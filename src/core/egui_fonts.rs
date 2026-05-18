// =============================================================================
// QubePixel — egui font configuration (TTF loading + fallback)
// =============================================================================

use crate::debug_log;

/// Configure egui fonts. Attempts to load a TTF from assets; falls back
/// to the built-in egui default (Roboto).
pub fn configure_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();

    let font_path = "assets/fonts/game_font.ttf";
    if std::path::Path::new(font_path).exists() {
        match std::fs::read(font_path) {
            Ok(data) => {
                fonts.font_data.insert(
                    "GameFont".to_owned(),
                    std::sync::Arc::new(egui::FontData::from_owned(data)),
                );

                // Set as primary proportional font
                fonts
                    .families
                    .entry(egui::FontFamily::Proportional)
                    .or_default()
                    .insert(0, "GameFont".to_owned());

                // Also as monospace fallback
                fonts
                    .families
                    .entry(egui::FontFamily::Monospace)
                    .or_default()
                    .insert(0, "GameFont".to_owned());

                ctx.set_fonts(fonts);
                debug_log!("EguiFonts", "configure_fonts",
                    "Loaded TTF font from {:?}", font_path);
            }
            Err(_e) => {
                // if IS_DEBUG {
                //     log::warn!(
                //         "[EguiFonts][configure_fonts] Failed to read {:?}: {}, \
                //          falling back to default egui font",
                //         font_path, e
                //     );
                // }
                // ctx.set_fonts(fonts);
            }
        }
    } else {
        debug_log!("EguiFonts", "configure_fonts",
            "Font file not found at {:?}, using default egui font", font_path);
        ctx.set_fonts(fonts);
    }
}