# Static obstacle prefabs

Drop `.glb`/`.usd` prop assets here (boxes, chairs, cones, whatever static
clutter should be scatterable in the hallway scene). `scene_randomizer`'s
`obstacle_randomizer.load_obstacle_prefab_library()` scans this directory at
runtime and treats every asset file in it as something that can be randomly
placed — there's no separate registry/manifest to keep in sync, so adding an
asset here is enough to make it available to the randomizer.

Keep this to reasonable static-obstacle-scale props (not full room/wall/ceiling
geometry) — the hallway shell itself is a separate fixed asset, not part of
this library.
