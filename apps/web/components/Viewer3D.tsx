"use client";

/**
 * Live 3D view of an IR.
 *
 * Geometry is built in the browser rather than fetched as a GLB. The server's
 * `/extrude` endpoint produces the durable artefact that the Pico headset
 * consumes; this rebuilds the same shapes locally so that editing a wall or
 * recalibrating the scale updates the view immediately instead of costing a
 * round trip. The two must agree, so both read the same IR and apply the same
 * plan -> world mapping:
 *
 *     plan (x, y)  ->  world (x, elevation, y)
 */

import { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Grid, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import {
  type FloorPlanIR,
  type Level,
  type Opening,
  type Room,
  type Wall,
  planBounds,
} from "@/lib/ir";

const COLORS = {
  exterior: "#c4beb4",
  interior: "#e0ddd6",
  floor: "#9e8d7a",
  selected: "#f59e0b",
} as const;

/**
 * Build one wall as a BoxGeometry, minus its openings.
 *
 * Openings are cut by rendering the wall as up to three stacked boxes around
 * each hole rather than by running a CSG boolean. For axis-aligned rectangular
 * openings the result is identical, and it avoids pulling a CSG library into
 * the client bundle for geometry that splits trivially. The server still does a
 * true boolean, because the exported GLB has to be watertight for VR.
 */
function wallSegments(
  wall: Wall,
  openings: Opening[],
  level: Level,
): Array<{ length: number; height: number; alongOffset: number; heightOffset: number }> {
  const height = wall.height ?? level.wallHeight;
  const length = Math.hypot(wall.end[0] - wall.start[0], wall.end[1] - wall.start[1]);

  if (openings.length === 0) {
    return [{ length, height, alongOffset: 0, heightOffset: height / 2 }];
  }

  const sorted = [...openings].sort((a, b) => a.offset - b.offset);
  const pieces: ReturnType<typeof wallSegments> = [];
  let cursor = 0;

  for (const opening of sorted) {
    const start = Math.max(0, opening.offset - opening.width / 2);
    const end = Math.min(length, opening.offset + opening.width / 2);

    // Full-height slab of wall before this opening.
    if (start > cursor + 1e-4) {
      const span = start - cursor;
      pieces.push({
        length: span,
        height,
        alongOffset: cursor + span / 2 - length / 2,
        heightOffset: height / 2,
      });
    }

    // Wall below the opening (a window's spandrel) and above it (the lintel).
    const spanWidth = end - start;
    if (spanWidth > 1e-4) {
      const center = start + spanWidth / 2 - length / 2;
      if (opening.sill > 1e-4) {
        pieces.push({
          length: spanWidth,
          height: opening.sill,
          alongOffset: center,
          heightOffset: opening.sill / 2,
        });
      }
      const lintelBase = opening.sill + opening.height;
      if (height - lintelBase > 1e-4) {
        pieces.push({
          length: spanWidth,
          height: height - lintelBase,
          alongOffset: center,
          heightOffset: lintelBase + (height - lintelBase) / 2,
        });
      }
    }

    cursor = Math.max(cursor, end);
  }

  if (length - cursor > 1e-4) {
    const span = length - cursor;
    pieces.push({
      length: span,
      height,
      alongOffset: cursor + span / 2 - length / 2,
      heightOffset: height / 2,
    });
  }

  return pieces;
}

function WallMesh({
  wall,
  openings,
  level,
  selected,
  onSelect,
}: {
  wall: Wall;
  openings: Opening[];
  level: Level;
  selected: boolean;
  onSelect?: (id: string) => void;
}) {
  const pieces = useMemo(() => wallSegments(wall, openings, level), [wall, openings, level]);

  const [x0, y0] = wall.start;
  const [x1, y1] = wall.end;
  const midX = (x0 + x1) / 2;
  const midY = (y0 + y1) / 2;
  // Rotating about +Y by phi sends local +X to (cos phi, 0, -sin phi); we need
  // it on (dx, 0, dy), hence the negated dy. Same derivation as the server.
  const yaw = Math.atan2(-(y1 - y0), x1 - x0);

  const color = selected
    ? COLORS.selected
    : wall.type === "exterior"
      ? COLORS.exterior
      : COLORS.interior;

  return (
    <group
      position={[midX, level.elevation, midY]}
      rotation={[0, yaw, 0]}
      onClick={(event) => {
        event.stopPropagation();
        onSelect?.(wall.id);
      }}
    >
      {pieces.map((piece, index) => (
        <mesh
          key={index}
          position={[piece.alongOffset, piece.heightOffset, 0]}
          castShadow
          receiveShadow
        >
          <boxGeometry args={[piece.length, piece.height, wall.thickness]} />
          <meshStandardMaterial color={color} roughness={0.85} metalness={0} />
        </mesh>
      ))}
    </group>
  );
}

function FloorMesh({ room, level }: { room: Room; level: Level }) {
  const geometry = useMemo(() => {
    const shape = new THREE.Shape();
    room.polygon.forEach(([x, y], index) => {
      // Shape is built in XY then laid flat by a -90 degree rotation about X,
      // which maps (x, y) -> (x, 0, -y). Negating y here cancels that, so the
      // floor lands in the same place as the walls above it.
      if (index === 0) shape.moveTo(x, -y);
      else shape.lineTo(x, -y);
    });
    shape.closePath();

    const geom = new THREE.ShapeGeometry(shape);
    geom.rotateX(-Math.PI / 2);
    return geom;
  }, [room.polygon]);

  return (
    <mesh geometry={geometry} position={[0, level.elevation - 0.01, 0]} receiveShadow>
      <meshStandardMaterial color={COLORS.floor} roughness={0.95} side={THREE.DoubleSide} />
    </mesh>
  );
}

function SpinLight() {
  const ref = useRef<THREE.DirectionalLight>(null);
  useFrame(({ clock }) => {
    // A slowly orbiting key light. Static lighting on untextured white massing
    // reads as flat; gentle movement makes the wall planes legible.
    const t = clock.getElapsedTime() * 0.1;
    ref.current?.position.set(Math.cos(t) * 14, 16, Math.sin(t) * 14);
  });
  return <directionalLight ref={ref} intensity={1.7} castShadow />;
}

export default function Viewer3D({
  ir,
  levelIndex = 0,
  selectedWallId,
  onSelectWall,
}: {
  ir: FloorPlanIR;
  levelIndex?: number;
  selectedWallId?: string | null;
  onSelectWall?: (id: string) => void;
}) {
  const level = ir.levels[levelIndex];

  const { center, size } = useMemo(() => planBounds(level), [level]);
  const span = Math.max(size[0], size[1], 4);

  const openingsByWall = useMemo(() => {
    const map = new Map<string, Opening[]>();
    for (const opening of level.openings) {
      const list = map.get(opening.wallId);
      if (list) list.push(opening);
      else map.set(opening.wallId, [opening]);
    }
    return map;
  }, [level.openings]);

  return (
    <Canvas
      shadows
      dpr={[1, 2]}
      camera={{ position: [center[0] + span * 0.9, span * 0.85, center[1] + span * 0.9], fov: 45 }}
      className="h-full w-full"
    >
      <color attach="background" args={["#11131a"]} />

      {/* Lighting is entirely local. drei's <Environment> would look nicer but
          it fetches an HDRI from a CDN and suspends the whole canvas until it
          arrives -- which renders a blank white viewport on a slow, offline, or
          firewalled network. The platform is destined for client-provided and
          potentially on-premise hosting, so the viewer must not depend on
          reaching the public internet to draw a frame. */}
      <hemisphereLight args={["#dfe7ff", "#40372c", 1.0]} />
      <ambientLight intensity={0.35} />
      <SpinLight />
      <directionalLight position={[-10, 9, -8]} intensity={0.45} />

      <group>
        {level.walls.map((wall) => (
          <WallMesh
            key={wall.id}
            wall={wall}
            openings={openingsByWall.get(wall.id) ?? []}
            level={level}
            selected={selectedWallId === wall.id}
            onSelect={onSelectWall}
          />
        ))}
        {level.rooms.map((room) => (
          <FloorMesh key={room.id} room={room} level={level} />
        ))}
      </group>

      <Grid
        args={[span * 4, span * 4]}
        position={[center[0], -0.02, center[1]]}
        cellSize={1}
        cellThickness={0.5}
        cellColor="#2a2f3d"
        sectionSize={5}
        sectionThickness={1}
        sectionColor="#3b4357"
        fadeDistance={span * 3.5}
        infiniteGrid={false}
      />

      <OrbitControls target={[center[0], 1.2, center[1]]} makeDefault enableDamping />
    </Canvas>
  );
}
