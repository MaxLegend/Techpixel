# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
cargo build --release   # always --release; debug is impractically slow
cargo run --release     # launches: main menu → game
cargo test              # CPU-logic tests; they live in src/core/world_gen/ and src/core/radiance_cascades/
cargo test <substr>     # run a single test (substring match on test name)
cargo clippy --release  # lint; output is sometimes captured to clippy_out.txt
```

## What is QubePixel?

Minecraft-style voxel engine in Rust. Features: real-time GI via **VCT** (Voxel Cone Tracing — 128³ flood-fill propagation), full biome terrain pipeline, fluid simulation (water/lava), skeletal player model, mobs (custom `EntityManager` + simple AI, e.g. sheep), inventory system, and in-game block/entity editors. Starts at 840×480, main menu → game. Config in `assets/lighting_config.json`, `assets/world_gen_config.json`, `assets/blocks/`, `assets/entities/`, `assets/chunk_config.json`.

## Conventions

- **Logging:** don't call `log::info!` directly for debug output. Use the macros `debug_log!`, `ext_debug_log!`, `flow_debug_log!` (defined in `src/core/logging.rs`), all in `[Class][Method] message` form, English only. They gate on the atomics `IS_DEBUG` / `IS_EXTENDED_DEBUG` / `IS_FLOW_DEBUG` respectively: wrap per-frame spam in `flow_debug_log!`, occasional diagnostics (mouse clicks, etc.) in `ext_debug_log!`, everything else in `debug_log!`.
- **Module layout:** `src/core/` = engine systems (world gen, lighting/VCT, fluid, entities, physics, save); `src/screens/` = render passes + the stack-based UI screens. Module visibility is declared in `src/core.rs` and `src/screens.rs`.

## Non-obvious invariants (read before touching any GPU code)

- **WGSL struct layout must byte-for-byte match Rust `bytemuck` types.** Compile-time size asserts in `system.rs` (both VCT and RC) enforce this — add one for any new GPU struct.
- Block ID `0xFFFF` in the RC voxel texture = unloaded air. Block ID `0` in the VCT snapshot = air.
- VCT volume is 128³ centered on camera. RC voxel texture is also 128³.
- Block LUT index 0 is reserved for air (fully transparent, no emission).

## Architecture — load what you need

| Task | Read |
|---|---|
| Entry point, event loop, screen stack | [ARCHITECTURE-ENTRY.md](ARCHITECTURE-ENTRY.md) |
| 3D render pipeline, frustum culling, camera, player model | [ARCHITECTURE-RENDERING.md](ARCHITECTURE-RENDERING.md) |
| Global illumination — VCT flood-fill + Radiance Cascades (archived) | [ARCHITECTURE-GI.md](ARCHITECTURE-GI.md) |
| Chunks, biome terrain gen, LOD, fluid simulation | [ARCHITECTURE-WORLD.md](ARCHITECTURE-WORLD.md) |
| Config atomics, lighting uniforms, physics, world gen config | [ARCHITECTURE-CONFIG.md](ARCHITECTURE-CONFIG.md) |
| Block JSON config — fields, placement/rotation modes, faces, emission | [BLOCKS-CONFIG-GUIDE.md](BLOCKS-CONFIG-GUIDE.md) |


Общие рабочие правила:
- Прежде чем приступать к выполнению, сформулируй короткий чеклист (3–7 пунктов) необходимых подзадач, чтобы структурировать выполнение.
- После ответа всегда проверяй, что ты не использовал несуществующие поля\метод\функции\переменные и что ты не создал неиспользуемые поля, методы, функции или переменные. 
- Код должен быть чистым, с комментариями к сложным участкам логики или математики.
- Добавляй логгирование основных действий с помощью log и env_logger в формате [Class][Method] - все логи на английском языке. 
- Все логи оборачивай в глобальную переменную IS_DEBUG. Логи потоковые (например те, которые выводятся каждый кадр - оборачивай в IS_FLOW_DEBUG. Те которые нужны не всегда (например логгирование кликов мыши) оборачивай в IS_EXT_DEBUG.  
- заверши ответ кратким абзацем, поясняющим, как твоё решение должно работать или выглядеть.
- Вместо поиска файлов по всей репозитории просто задавай вопрос, где лежит файл, если не знаешь.
- В конце реализации задачи сообщай, как протестировать нововведения или исправления.
- **Всегда отвечай на русском языке.**

