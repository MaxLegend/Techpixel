// =============================================================================
// QubePixel — Inventory System (Component + System)
// =============================================================================

use std::collections::HashMap;
use egui::{Color32, Pos2, Rect, Rounding, Stroke, Vec2, Align2, FontId, Margin};

use crate::screens::inventory_model_renderer::ModelPreviewSlot;
use crate::{debug_log, ext_debug_log, flow_debug_log};

// =============================================================================
// Constants
// =============================================================================

const SLOT_SIZE: f32 = 48.0;
const SLOT_GAP: f32  = 4.0;
const MAIN_COLS: usize = 9;

// =============================================================================
// Color palette
// =============================================================================

mod palette {
    use egui::Color32;
    pub fn slot_bg()          -> Color32 { Color32::from_rgba_unmultiplied(55,  55,  55,  220) }
    pub fn slot_active()      -> Color32 { Color32::from_rgba_unmultiplied(80,  80,  80,  240) }
    pub fn slot_border_dark() -> Color32 { Color32::from_rgba_unmultiplied(20,  20,  20,  255) }
    pub fn slot_border_lit()  -> Color32 { Color32::from_rgba_unmultiplied(110, 110, 110, 255) }
    pub fn window_bg()        -> Color32 { Color32::from_rgba_unmultiplied(35,  35,  35,  245) }
    pub fn window_border()    -> Color32 { Color32::from_rgb(20, 20, 20) }
    pub fn text_primary()     -> Color32 { Color32::WHITE }
    pub fn text_count()       -> Color32 { Color32::WHITE }
    pub fn hotbar_bg()        -> Color32 { Color32::from_rgba_unmultiplied(25,  25,  25,  215) }
    pub fn mode_creative()    -> Color32 { Color32::from_rgb(100, 220, 255) }
    pub fn mode_survival()    -> Color32 { Color32::from_rgb(255, 210, 100) }
    pub fn tab_active_bg()    -> Color32 { Color32::from_rgba_unmultiplied(50,  90,  130, 240) }
    pub fn tab_active_text()  -> Color32 { Color32::from_rgb(120, 220, 255) }
    pub fn tab_idle_text()    -> Color32 { Color32::from_rgba_unmultiplied(180, 180, 180, 230) }
    pub fn section_label()    -> Color32 { Color32::from_rgba_unmultiplied(160, 160, 160, 180) }
}

// =============================================================================
// COMPONENT LAYER — Data Structures
// =============================================================================

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ItemStack {
    pub block_id:    String,
    pub display_name: String,
    pub amount:      u32,
    pub max_stack:   u32,
    pub block_color: [f32; 3],
    #[serde(default)]
    pub top_texture:  Option<String>,
    #[serde(default)]
    pub side_texture: Option<String>,
    /// Creative-mode tab this item belongs to (mirrors BlockDefinition.inventory_tab).
    #[serde(default)]
    pub inventory_tab: Option<String>,
    /// If this block has a custom 3D model, the model ID (e.g. "block/redstone_furnace_je").
    #[serde(default)]
    pub model_name: Option<String>,
}

impl ItemStack {
    pub fn new(block_id: impl Into<String>, display_name: impl Into<String>,
               amount: u32, max_stack: u32) -> Self {
        let amount = amount.min(max_stack);
        Self {
            block_id: block_id.into(), display_name: display_name.into(),
            amount, max_stack, block_color: [0.5, 0.5, 0.5],
            top_texture: None, side_texture: None, inventory_tab: None,
            model_name: None,
        }
    }

    pub fn with_color(mut self, color: [f32; 3]) -> Self { self.block_color = color; self }

    pub fn with_textures(mut self, top: Option<String>, side: Option<String>) -> Self {
        self.top_texture  = top;
        self.side_texture = side;
        self
    }

    pub fn with_tab(mut self, tab: Option<String>) -> Self {
        self.inventory_tab = tab;
        self
    }

    pub fn with_model(mut self, model_name: Option<String>) -> Self {
        self.model_name = model_name;
        self
    }

    pub fn single(block_id: impl Into<String>, display_name: impl Into<String>,
                  max_stack: u32) -> Self {
        Self::new(block_id, display_name, 1, max_stack)
    }

    #[inline] pub fn is_full(&self)  -> bool { self.amount >= self.max_stack }
    #[inline] pub fn is_empty(&self) -> bool { self.amount == 0 }

    #[inline]
    pub fn can_merge(&self, other: &ItemStack) -> bool {
        self.block_id == other.block_id && !self.is_full()
    }

    pub fn merge_from(&mut self, other: ItemStack) -> Option<ItemStack> {
        if self.block_id != other.block_id { return Some(other); }
        let space = self.max_stack - self.amount;
        if other.amount <= space {
            self.amount += other.amount;
            None
        } else {
            self.amount = self.max_stack;
            let remainder = other.amount - space;
            Some(ItemStack { amount: remainder, ..other })
        }
    }

    pub fn split_half(&mut self) -> Option<ItemStack> {
        if self.amount <= 1 { return None; }
        let half       = self.amount / 2;
        let other_half = self.amount - half;
        self.amount    = half;
        Some(ItemStack {
            block_id:    self.block_id.clone(),
            display_name: self.display_name.clone(),
            amount:      other_half,
            max_stack:   self.max_stack,
            block_color: self.block_color,
            top_texture:  self.top_texture.clone(),
            side_texture: self.side_texture.clone(),
            inventory_tab: self.inventory_tab.clone(),
            model_name:  self.model_name.clone(),
        })
    }

    pub fn consume(&mut self, count: u32) -> bool {
        if self.amount < count { return false; }
        self.amount -= count;
        true
    }
}

// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum InventorySlot {
    Empty,
    Occupied(ItemStack),
}

impl InventorySlot {
    pub fn empty() -> Self { InventorySlot::Empty }
    pub fn occupied(item: ItemStack) -> Self { InventorySlot::Occupied(item) }
    #[inline] pub fn is_empty(&self)    -> bool { matches!(self, InventorySlot::Empty) }
    #[inline] pub fn is_occupied(&self) -> bool { matches!(self, InventorySlot::Occupied(_)) }

    pub fn item(&self) -> Option<&ItemStack> {
        match self { InventorySlot::Empty => None, InventorySlot::Occupied(i) => Some(i) }
    }

    pub fn take(&mut self) -> Option<ItemStack> {
        match std::mem::replace(self, InventorySlot::Empty) {
            InventorySlot::Empty        => None,
            InventorySlot::Occupied(it) => Some(it),
        }
    }

    pub fn take_half(&mut self) -> Option<ItemStack> {
        match self {
            InventorySlot::Empty        => None,
            InventorySlot::Occupied(it) => it.split_half(),
        }
    }

    pub fn insert(&mut self, stack: ItemStack) -> Option<ItemStack> {
        match self {
            InventorySlot::Empty           => { *self = InventorySlot::Occupied(stack); None }
            InventorySlot::Occupied(exist) => exist.merge_from(stack),
        }
    }

    pub fn place_one(&mut self, item: &ItemStack) -> bool {
        match self {
            InventorySlot::Empty => {
                *self = InventorySlot::Occupied(ItemStack {
                    block_id:    item.block_id.clone(),
                    display_name: item.display_name.clone(),
                    amount:      1,
                    max_stack:   item.max_stack,
                    block_color: item.block_color,
                    top_texture:  item.top_texture.clone(),
                    side_texture: item.side_texture.clone(),
                    inventory_tab: item.inventory_tab.clone(),
                    model_name:  item.model_name.clone(),
                });
                true
            }
            InventorySlot::Occupied(exist) => {
                if exist.can_merge(item) { exist.amount += 1; true } else { false }
            }
        }
    }

    pub fn swap(a: &mut InventorySlot, b: &mut InventorySlot) { std::mem::swap(a, b); }
}

impl Default for InventorySlot {
    fn default() -> Self { InventorySlot::Empty }
}

// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotTarget {
    Main(usize),
    Hotbar(usize),
    Creative(usize),
    Trash,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PlayerInventory {
    pub main_slots:           Vec<InventorySlot>,
    pub hotbar_slots:         Vec<InventorySlot>,
    pub active_hotbar_index:  usize,
    pub dragged_item:         Option<ItemStack>,
    pub is_open:              bool,
    pub creative_mode:        bool,
}

impl PlayerInventory {
    pub fn new(main_count: usize, hotbar_count: usize) -> Self {
        debug_log!("PlayerInventory", "new",
            "Creating inventory: main={} hotbar={}", main_count, hotbar_count);
        Self {
            main_slots:          vec![InventorySlot::Empty; main_count],
            hotbar_slots:        vec![InventorySlot::Empty; hotbar_count],
            active_hotbar_index: 0,
            dragged_item:        None,
            is_open:             false,
            creative_mode:       false,
        }
    }

    pub fn main_slot(&self,     i: usize) -> Option<&InventorySlot>     { self.main_slots.get(i) }
    pub fn main_slot_mut(&mut self, i: usize) -> Option<&mut InventorySlot> { self.main_slots.get_mut(i) }
    pub fn hotbar_slot(&self,   i: usize) -> Option<&InventorySlot>     { self.hotbar_slots.get(i) }
    pub fn hotbar_slot_mut(&mut self, i: usize) -> Option<&mut InventorySlot> { self.hotbar_slots.get_mut(i) }

    pub fn slot_mut_by_target(&mut self, target: SlotTarget) -> Option<&mut InventorySlot> {
        match target {
            SlotTarget::Main(i)    => self.main_slot_mut(i),
            SlotTarget::Hotbar(i)  => self.hotbar_slot_mut(i),
            SlotTarget::Creative(_) | SlotTarget::Trash => None,
        }
    }

    pub fn active_item(&self) -> Option<&ItemStack> {
        self.hotbar_slots.get(self.active_hotbar_index).and_then(|s| s.item())
    }

    pub fn active_block_id(&self)   -> Option<&str> { self.active_item().map(|i| i.block_id.as_str()) }
    pub fn active_block_name(&self) -> &str {
        self.active_item().map(|i| i.display_name.as_str()).unwrap_or("Empty")
    }

    pub fn hotbar_count(&self) -> usize { self.hotbar_slots.len() }

    pub fn use_active_block(&mut self) -> Option<(String, String, bool)> {
        let idx  = self.active_hotbar_index;
        let info = {
            let slot = self.hotbar_slots.get(idx)?;
            let item = slot.item()?;
            (item.block_id.clone(), item.display_name.clone())
        };
        if !self.creative_mode {
            let slot = &mut self.hotbar_slots[idx];
            if let InventorySlot::Occupied(item) = slot {
                item.consume(1);
                if item.is_empty() { *slot = InventorySlot::Empty; }
            }
            Some((info.0, info.1, true))
        } else {
            Some((info.0, info.1, false))
        }
    }

    pub fn scroll_hotbar(&mut self, delta: i32) {
        if self.hotbar_slots.is_empty() { return; }
        let n = self.hotbar_slots.len();
        let d = delta.rem_euclid(n as i32) as usize;
        self.active_hotbar_index = (self.active_hotbar_index + d) % n;
    }

    pub fn set_active_hotbar(&mut self, index: usize) {
        if !self.hotbar_slots.is_empty() && index < self.hotbar_slots.len() {
            self.active_hotbar_index = index;
        }
    }

    /// Clear (delete) the hotbar slot at `idx`.
    pub fn clear_hotbar_slot(&mut self, idx: usize) {
        if let Some(slot) = self.hotbar_slots.get_mut(idx) {
            *slot = InventorySlot::Empty;
            debug_log!("PlayerInventory", "clear_hotbar_slot", "Cleared hotbar slot {}", idx);
        }
    }

    pub fn add_item(&mut self, mut stack: ItemStack) -> Option<ItemStack> {
        if stack.is_empty() { return None; }
        for slot in &mut self.hotbar_slots {
            if let Some(r) = slot.insert(stack) { stack = r; } else { return None; }
        }
        for slot in &mut self.main_slots {
            if let Some(r) = slot.insert(stack) { stack = r; } else { return None; }
        }
        for slot in &mut self.hotbar_slots {
            if slot.is_empty() { slot.insert(stack); return None; }
        }
        for slot in &mut self.main_slots {
            if slot.is_empty() { slot.insert(stack); return None; }
        }
        debug_log!("PlayerInventory", "add_item",
            "Inventory full: {} x{}", stack.block_id, stack.amount);
        Some(stack)
    }

    pub fn toggle_open(&mut self) {
        self.is_open = !self.is_open;
        if !self.is_open {
            if let Some(item) = self.dragged_item.take() {
                debug_log!("PlayerInventory", "toggle_open",
                    "Returning dragged: {} x{}", item.block_id, item.amount);
                self.add_item(item);
            }
        }
    }

    pub fn hotbar_slot_name(&self, i: usize) -> &str {
        self.hotbar_slot(i).and_then(|s| s.item())
            .map(|i| i.display_name.as_str()).unwrap_or("Empty")
    }
}

// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct RenderCommand {
    pub rect:     Rect,
    pub block_id: String,
    pub count:    u32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InventoryConfig {
    pub main_slots_count:    usize,
    pub hotbar_slots_count:  usize,
    pub creative_mode:       bool,
    pub default_max_stack:   u32,
    pub max_stack_sizes:     HashMap<String, u32>,
}

impl Default for InventoryConfig {
    fn default() -> Self {
        Self {
            main_slots_count:   27,
            hotbar_slots_count: 12,
            creative_mode:      true,
            default_max_stack:  64,
            max_stack_sizes:    HashMap::new(),
        }
    }
}

impl InventoryConfig {
    pub fn load(path: &str) -> Self {
        debug_log!("InventoryConfig", "load", "Loading config from: {}", path);
        match std::fs::read_to_string(path) {
            Ok(data) => match serde_json::from_str::<InventoryConfig>(&data) {
                Ok(cfg) => cfg,
                Err(e)  => {
                    debug_log!("InventoryConfig", "load", "Parse error ({}), using defaults", e);
                    InventoryConfig::default()
                }
            },
            Err(_) => InventoryConfig::default(),
        }
    }

    pub fn get_max_stack(&self, block_id: &str) -> u32 {
        self.max_stack_sizes.get(block_id).copied().unwrap_or(self.default_max_stack)
    }
}

// =============================================================================
// UI SYSTEM LAYER — InventoryUI
// =============================================================================

#[derive(Debug, Clone, Copy)]
enum PendingClick {
    Left(SlotTarget),
    Right(SlotTarget),
    Delete(SlotTarget),
}

pub struct InventoryUI {
    pub inventory:        PlayerInventory,
    pub config:           InventoryConfig,
    creative_items:       Vec<ItemStack>,
    last_render_commands: Vec<RenderCommand>,
    inventory_interacted: bool,
    creative_mode:        bool,
    tile_images:          HashMap<String, egui::ColorImage>,
    tile_textures:        HashMap<String, egui::TextureHandle>,
    /// Tab list: index 0 is always "All". Rest are sorted unique tab names from items.
    available_tabs:       Vec<String>,
    /// Currently selected tab index (0 = "All").
    selected_tab:         usize,
    /// Model preview slots collected during build_ui for post-egui rendering.
    model_preview_slots:  Vec<ModelPreviewSlot>,
}

impl InventoryUI {
    pub fn new(config: InventoryConfig) -> Self {
        debug_log!("InventoryUI", "new",
            "Creating InventoryUI: main={} hotbar={}", config.main_slots_count, config.hotbar_slots_count);
        let creative_mode = config.creative_mode;
        let mut inventory = PlayerInventory::new(config.main_slots_count, config.hotbar_slots_count);
        inventory.creative_mode = creative_mode;
        Self {
            inventory, config,
            creative_items:       Vec::new(),
            last_render_commands: Vec::new(),
            inventory_interacted: false,
            creative_mode,
            tile_images:          HashMap::new(),
            tile_textures:        HashMap::new(),
            available_tabs:       vec!["All".to_string()],
            selected_tab:         0,
            model_preview_slots:  Vec::new(),
        }
    }

    pub fn add_tile_image(&mut self, name: String, image: egui::ColorImage) {
        self.tile_images.entry(name).or_insert(image);
    }

    fn ensure_tile_texture(&mut self, name: &str, ctx: &egui::Context) {
        if self.tile_textures.contains_key(name) { return; }
        if let Some(ci) = self.tile_images.remove(name) {
            let handle = ctx.load_texture(
                name, ci,
                egui::TextureOptions {
                    magnification: egui::TextureFilter::Nearest,
                    minification:  egui::TextureFilter::Nearest,
                    ..Default::default()
                },
            );
            self.tile_textures.insert(name.to_string(), handle);
        }
    }

    /// Rebuild the tab list from the current creative items.
    fn rebuild_tabs(&mut self) {
        let mut tab_set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for item in &self.creative_items {
            if let Some(tab) = &item.inventory_tab {
                if !tab.trim().is_empty() {
                    tab_set.insert(tab.clone());
                }
            }
        }
        self.available_tabs = std::iter::once("All".to_string())
            .chain(tab_set.into_iter())
            .collect();
        self.selected_tab = 0;
        debug_log!("InventoryUI", "rebuild_tabs",
            "Tabs built: {:?}", self.available_tabs);
    }

    pub fn set_creative_items(&mut self, items: Vec<ItemStack>) {
        debug_log!("InventoryUI", "set_creative_items",
            "Creative palette: {} block types", items.len());
        self.creative_items = items;
        self.rebuild_tabs();
    }

    pub fn toggle_creative(&mut self) {
        self.creative_mode         = !self.creative_mode;
        self.inventory.creative_mode = self.creative_mode;
        debug_log!("InventoryUI", "toggle_creative", "Creative mode: {}", self.creative_mode);
    }

    pub fn is_creative(&self)        -> bool { self.creative_mode }
    pub fn toggle(&mut self)          { self.inventory.toggle_open(); }
    pub fn is_open(&self)             -> bool { self.inventory.is_open }
    pub fn was_interacted(&self)      -> bool { self.inventory_interacted }
    pub fn active_block_id(&self)     -> Option<&str> { self.inventory.active_block_id() }
    pub fn active_block_name(&self)   -> &str { self.inventory.active_block_name() }
    pub fn hotbar_count(&self)        -> usize { self.inventory.hotbar_count() }
    pub fn hotbar_slot_name(&self, i: usize) -> &str { self.inventory.hotbar_slot_name(i) }
    pub fn render_commands(&self)     -> &[RenderCommand] { &self.last_render_commands }

    pub fn take_model_preview_slots(&mut self) -> Vec<ModelPreviewSlot> {
        std::mem::take(&mut self.model_preview_slots)
    }

    pub fn use_active_block(&mut self) -> Option<(String, String, bool)> {
        self.inventory.use_active_block()
    }

    pub fn scroll_hotbar(&mut self, delta: i32) -> bool {
        if self.inventory.is_open { return false; }
        self.inventory.scroll_hotbar(delta);
        true
    }

    pub fn select_hotbar(&mut self, index: usize) -> bool {
        if self.inventory.is_open { return false; }
        self.inventory.set_active_hotbar(index);
        true
    }

    /// Clear the currently-active hotbar slot (Q-drop / player requested delete).
    pub fn clear_active_hotbar_slot(&mut self) {
        let idx = self.inventory.active_hotbar_index;
        self.inventory.clear_hotbar_slot(idx);
    }

    // ----- Slot interaction -------------------------------------------------

    fn handle_left_click(&mut self, target: SlotTarget) {
        ext_debug_log!("InventoryUI", "handle_left_click", "Target: {:?}", target);

        if let SlotTarget::Creative(idx) = target {
            self.inventory.dragged_item = None;
            if let Some(tpl) = self.creative_items.get(idx) {
                let item = ItemStack::new(
                    tpl.block_id.clone(), tpl.display_name.clone(),
                    tpl.max_stack, tpl.max_stack,
                ).with_color(tpl.block_color)
                 .with_textures(tpl.top_texture.clone(), tpl.side_texture.clone())
                 .with_tab(tpl.inventory_tab.clone());
                let ai = self.inventory.active_hotbar_index;
                if let Some(slot) = self.inventory.hotbar_slot_mut(ai) {
                    *slot = InventorySlot::Occupied(item);
                }
            }
            return;
        }

        if let SlotTarget::Trash = target {
            self.inventory.dragged_item = None;
            return;
        }

        if self.inventory.dragged_item.is_none() {
            if let Some(slot) = self.inventory.slot_mut_by_target(target) {
                let taken = slot.take();
                if let Some(item) = taken { self.inventory.dragged_item = Some(item); }
            }
        } else {
            let dragged = self.inventory.dragged_item.take().unwrap();
            if let Some(slot) = self.inventory.slot_mut_by_target(target) {
                match slot.insert(dragged) {
                    None           => {}
                    Some(remainder) => {
                        let old = slot.take().unwrap();
                        *slot = InventorySlot::Occupied(remainder);
                        self.inventory.dragged_item = Some(old);
                    }
                }
            } else {
                self.inventory.dragged_item = Some(dragged);
            }
        }
    }

    fn handle_right_click(&mut self, target: SlotTarget) {
        ext_debug_log!("InventoryUI", "handle_right_click", "Target: {:?}", target);

        if let SlotTarget::Creative(idx) = target {
            self.inventory.dragged_item = None;
            if let Some(tpl) = self.creative_items.get(idx) {
                let item = ItemStack::single(
                    tpl.block_id.clone(), tpl.display_name.clone(), tpl.max_stack,
                ).with_color(tpl.block_color)
                 .with_textures(tpl.top_texture.clone(), tpl.side_texture.clone())
                 .with_tab(tpl.inventory_tab.clone());
                let ai = self.inventory.active_hotbar_index;
                if let Some(slot) = self.inventory.hotbar_slot_mut(ai) {
                    *slot = InventorySlot::Occupied(item);
                }
            }
            return;
        }

        if let SlotTarget::Trash = target {
            self.inventory.dragged_item = None;
            return;
        }

        if self.inventory.dragged_item.is_none() {
            if let Some(slot) = self.inventory.slot_mut_by_target(target) {
                let half = slot.take_half();
                if let Some(item) = half { self.inventory.dragged_item = Some(item); }
            }
        } else {
            let dragged_info = self.inventory.dragged_item.as_ref().map(|d| {
                (d.block_id.clone(), d.display_name.clone(), d.max_stack)
            });
            if let Some((bid, dn, ms)) = dragged_info {
                if let Some(slot) = self.inventory.slot_mut_by_target(target) {
                    let tpl = ItemStack::single(bid, dn, ms);
                    if slot.place_one(&tpl) {
                        if let Some(ref mut d) = self.inventory.dragged_item {
                            d.consume(1);
                            if d.is_empty() { self.inventory.dragged_item = None; }
                        }
                    }
                }
            }
        }
    }

    fn handle_delete(&mut self, target: SlotTarget) {
        ext_debug_log!("InventoryUI", "handle_delete", "Target: {:?}", target);
        match target {
            SlotTarget::Hotbar(i) => self.inventory.clear_hotbar_slot(i),
            SlotTarget::Main(i)   => {
                if let Some(slot) = self.inventory.main_slot_mut(i) {
                    *slot = InventorySlot::Empty;
                }
            }
            SlotTarget::Trash => { self.inventory.dragged_item = None; }
            _ => {}
        }
    }

    fn process_pending_clicks(&mut self, clicks: Vec<PendingClick>) {
        for click in clicks {
            match click {
                PendingClick::Left(t)   => self.handle_left_click(t),
                PendingClick::Right(t)  => self.handle_right_click(t),
                PendingClick::Delete(t) => self.handle_delete(t),
            }
        }
    }

    // ----- egui rendering ---------------------------------------------------

    pub fn build_ui(&mut self, ctx: &egui::Context) -> Vec<RenderCommand> {
        self.last_render_commands.clear();
        self.inventory_interacted = false;
        self.model_preview_slots.clear();

        self.build_hotbar(ctx);

        if self.inventory.is_open {
            self.build_inventory_window(ctx);
        }

        self.draw_dragged_item(ctx);

        std::mem::take(&mut self.last_render_commands)
    }

    fn draw_slot_block(
        &mut self,
        painter: &egui::Painter,
        rect: Rect,
        color: [f32; 3],
        top_tex: Option<egui::TextureId>,
        side_tex: Option<egui::TextureId>,
        model_name: Option<&str>,
    ) {
        if let Some(mn) = model_name {
            self.model_preview_slots.push(ModelPreviewSlot {
                model_id: mn.to_string(),
                rect,
            });
            draw_block_iso_placeholder(painter, rect, color);
        } else {
            draw_block_iso(painter, rect, color, top_tex, side_tex);
        }
    }

    // ----- Hotbar -----------------------------------------------------------

    fn build_hotbar(&mut self, ctx: &egui::Context) {
        let screen_rect = ctx.screen_rect();
        let sw = screen_rect.width();
        let sh = screen_rect.height();
        let n  = self.inventory.hotbar_count();
        if n == 0 { return; }

        struct SlotSnap {
            block_color:  [f32; 3],
            amount:       u32,
            block_id:     String,
            top_texture:  Option<String>,
            side_texture: Option<String>,
            model_name:   Option<String>,
        }

        let active_idx = self.inventory.active_hotbar_index;
        let slots: Vec<(bool, Option<SlotSnap>)> = (0..n).map(|i| {
            let is_active = i == active_idx;
            let snap = self.inventory.hotbar_slot(i)
                .and_then(|s| s.item())
                .map(|it| SlotSnap {
                    block_color:  it.block_color,
                    amount:       it.amount,
                    block_id:     it.block_id.clone(),
                    top_texture:  it.top_texture.clone(),
                    side_texture: it.side_texture.clone(),
                    model_name:   it.model_name.clone(),
                });
            (is_active, snap)
        }).collect();

        for (_, snap) in &slots {
            if let Some(s) = snap {
                if let Some(ref nm) = s.top_texture  { self.ensure_tile_texture(nm, ctx); }
                if let Some(ref nm) = s.side_texture { self.ensure_tile_texture(nm, ctx); }
            }
        }

        let total_w = n as f32 * SLOT_SIZE + (n - 1) as f32 * SLOT_GAP;
        let start_x = (sw - total_w) / 2.0;
        let start_y = sh - SLOT_SIZE - 8.0;

        let layer_id = egui::LayerId::new(egui::Order::Foreground, egui::Id::new("hotbar_layer"));
        let painter  = ctx.layer_painter(layer_id);

        let pad    = 6.0;
        let bg_rect = Rect::from_min_size(
            Pos2::new(start_x - pad, start_y - pad),
            Vec2::new(total_w + pad * 2.0, SLOT_SIZE + pad * 2.0),
        );
        painter.rect_filled(bg_rect, Rounding::ZERO, palette::hotbar_bg());
        let (tl, tr, bl, br) = (bg_rect.left_top(), bg_rect.right_top(),
                                 bg_rect.left_bottom(), bg_rect.right_bottom());
        painter.line_segment([tl, tr], Stroke::new(2.0, Color32::from_white_alpha(40)));
        painter.line_segment([tl, bl], Stroke::new(2.0, Color32::from_white_alpha(40)));
        painter.line_segment([bl, br], Stroke::new(2.0, Color32::from_black_alpha(140)));
        painter.line_segment([tr, br], Stroke::new(2.0, Color32::from_black_alpha(140)));

        let mode_text  = if self.creative_mode { "CREATIVE" } else { "SURVIVAL" };
        let mode_color = if self.creative_mode { palette::mode_creative() } else { palette::mode_survival() };
        painter.text(
            Pos2::new(start_x - pad, start_y - pad - 2.0), Align2::LEFT_BOTTOM,
            mode_text, FontId::monospace(10.0), mode_color,
        );

        for (i, (is_active, snap)) in slots.iter().enumerate() {
            let x = start_x + i as f32 * (SLOT_SIZE + SLOT_GAP);
            let slot_rect = Rect::from_min_size(Pos2::new(x, start_y), Vec2::splat(SLOT_SIZE));

            draw_slot_bg(&painter, slot_rect, *is_active);
            if *is_active {
                painter.rect_stroke(
                    slot_rect.expand(2.0), Rounding::ZERO,
                    Stroke::new(2.0, Color32::WHITE), egui::StrokeKind::Outside,
                );
            }
            painter.text(
                slot_rect.left_top() + Vec2::new(2.0, 1.0), Align2::LEFT_TOP,
                &(i + 1).to_string(), FontId::monospace(8.0),
                Color32::from_white_alpha(if *is_active { 180 } else { 80 }),
            );

            if let Some(s) = snap {
                let top_tex  = s.top_texture.as_ref().and_then(|nm| self.tile_textures.get(nm).map(|h| h.id()));
                let side_tex = s.side_texture.as_ref().and_then(|nm| self.tile_textures.get(nm).map(|h| h.id()));
                self.draw_slot_block(&painter, slot_rect, s.block_color, top_tex, side_tex, s.model_name.as_deref());

                if s.amount > 1 {
                    let pos = slot_rect.right_bottom() + Vec2::new(-3.0, -1.0);
                    painter.text(pos + Vec2::new(1.0, 1.0), Align2::RIGHT_BOTTOM,
                        &s.amount.to_string(), FontId::monospace(11.0), Color32::from_black_alpha(200));
                    painter.text(pos, Align2::RIGHT_BOTTOM,
                        &s.amount.to_string(), FontId::monospace(11.0), palette::text_count());
                }
                self.last_render_commands.push(RenderCommand {
                    rect: slot_rect, block_id: s.block_id.clone(), count: s.amount,
                });
            }
        }

        if !self.inventory.is_open {
            painter.text(
                Pos2::new(sw / 2.0, sh - 2.0), Align2::CENTER_BOTTOM,
                "Scroll / 1-9 select  |  E inventory  |  Q drop slot  |  C mode",
                FontId::monospace(9.0), Color32::from_rgba_unmultiplied(160, 160, 160, 80),
            );
        }
    }

    // ----- Inventory window -------------------------------------------------

    fn build_inventory_window(&mut self, ctx: &egui::Context) {
        let mut open = true;
        let mut pending_clicks: Vec<PendingClick> = Vec::new();
        let mut pending_tab_change: Option<usize>  = None;

        // ── Snapshot read-only state for use inside closure ──────────────────
        let selected_tab     = self.selected_tab;
        let available_tabs   = self.available_tabs.clone();
        let filter_tab_name  = available_tabs.get(selected_tab).cloned().unwrap_or_default();
        let show_all_tabs    = selected_tab == 0;

        let title      = if self.creative_mode { "CREATIVE INVENTORY" } else { "INVENTORY" };
        let title_color = if self.creative_mode { palette::mode_creative() } else { palette::text_primary() };

        // ── Preload textures ─────────────────────────────────────────────────
        {
            let mut names: Vec<String> = Vec::new();
            for slot in &self.inventory.main_slots {
                if let Some(it) = slot.item() {
                    if let Some(n) = &it.top_texture  { names.push(n.clone()); }
                    if let Some(n) = &it.side_texture { names.push(n.clone()); }
                }
            }
            let hc = self.inventory.hotbar_count();
            for i in 0..hc {
                if let Some(it) = self.inventory.hotbar_slot(i).and_then(|s| s.item()) {
                    if let Some(n) = &it.top_texture  { names.push(n.clone()); }
                    if let Some(n) = &it.side_texture { names.push(n.clone()); }
                }
            }
            for it in &self.creative_items {
                if let Some(n) = &it.top_texture  { names.push(n.clone()); }
                if let Some(n) = &it.side_texture { names.push(n.clone()); }
            }
            if let Some(ref d) = self.inventory.dragged_item {
                if let Some(n) = &d.top_texture  { names.push(n.clone()); }
                if let Some(n) = &d.side_texture { names.push(n.clone()); }
            }
            for name in names { self.ensure_tile_texture(&name, ctx); }
        }

        let window = egui::Window::new("Inventory")
            .title_bar(false)
            .resizable(false)
            .movable(false)
            .collapsible(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 20.0])
            .frame(egui::Frame {
                fill:         palette::window_bg(),
                corner_radius: Rounding::ZERO,
                stroke:       Stroke::new(2.0, palette::window_border()),
                inner_margin: Margin::same(12),
                ..Default::default()
            })
            .open(&mut open);

        window.show(ctx, |ui| {
            ui.visuals_mut().widgets.inactive.corner_radius = Rounding::ZERO;
            ui.visuals_mut().widgets.active.corner_radius   = Rounding::ZERO;
            ui.visuals_mut().widgets.hovered.corner_radius  = Rounding::ZERO;

            ui.vertical_centered(|ui| {
                ui.label(egui::RichText::new(title).size(16.0).color(title_color).monospace());
                ui.add_space(6.0);

                // ── Main grid (survival only) ────────────────────────────────
                if !self.creative_mode {
                    let main_rows = (self.inventory.main_slots.len() + MAIN_COLS - 1) / MAIN_COLS;
                    egui::Grid::new("main_inventory_grid")
                        .spacing(Vec2::new(SLOT_GAP, SLOT_GAP))
                        .min_col_width(SLOT_SIZE)
                        .max_col_width(SLOT_SIZE)
                        .show(ui, |ui| {
                            for row in 0..main_rows {
                                for col in 0..MAIN_COLS {
                                    let idx = row * MAIN_COLS + col;
                                    let (bid, amount, color, top_nm, side_nm, mn) =
                                        self.inventory.main_slot(idx)
                                            .and_then(|s| s.item())
                                            .map(|i| (i.block_id.clone(), i.amount, i.block_color,
                                                      i.top_texture.clone(), i.side_texture.clone(),
                                                      i.model_name.clone()))
                                            .unwrap_or_default();

                                    let resp = ui.add_sized(
                                        [SLOT_SIZE, SLOT_SIZE],
                                        egui::Button::new("").fill(palette::slot_bg())
                                            .stroke(Stroke::new(2.0, palette::slot_border_dark())),
                                    );
                                    if resp.clicked()            { pending_clicks.push(PendingClick::Left(SlotTarget::Main(idx)));   self.inventory_interacted = true; }
                                    else if resp.secondary_clicked() { pending_clicks.push(PendingClick::Right(SlotTarget::Main(idx))); self.inventory_interacted = true; }
                                    else if resp.middle_clicked() { pending_clicks.push(PendingClick::Delete(SlotTarget::Main(idx))); self.inventory_interacted = true; }

                                    if !bid.is_empty() {
                                        let top_tex  = top_nm.as_ref().and_then(|n| self.tile_textures.get(n).map(|h| h.id()));
                                        let side_tex = side_nm.as_ref().and_then(|n| self.tile_textures.get(n).map(|h| h.id()));
                                        self.draw_slot_block(ui.painter(), resp.rect, color, top_tex, side_tex, mn.as_deref());
                                        if amount > 1 {
                                            let p   = ui.painter();
                                            let pos = resp.rect.right_bottom() + Vec2::new(-3.0, -2.0);
                                            p.text(pos + Vec2::new(1.0, 1.0), Align2::RIGHT_BOTTOM, &amount.to_string(), FontId::monospace(11.0), Color32::from_black_alpha(200));
                                            p.text(pos, Align2::RIGHT_BOTTOM, &amount.to_string(), FontId::monospace(11.0), palette::text_count());
                                        }
                                        self.last_render_commands.push(RenderCommand { rect: resp.rect, block_id: bid, count: amount });
                                    }
                                    if resp.hovered() && self.inventory.main_slot(idx).and_then(|s| s.item()).is_some() {
                                        let name = self.inventory.main_slot(idx).and_then(|s| s.item())
                                            .map(|i| i.display_name.as_str()).unwrap_or("");
                                        egui::show_tooltip_text(ctx, ui.layer_id(),
                                            egui::Id::new(("main_tip", idx)), name);
                                    }
                                }
                                ui.end_row();
                            }
                        });

                    ui.add_space(8.0);
                    ui.add(egui::Separator::default().horizontal().spacing(4.0));
                    ui.add_space(4.0);
                }

                // ── Creative panel ───────────────────────────────────────────
                if self.creative_mode && !self.creative_items.is_empty() {
                    // ── Tab bar ──────────────────────────────────────────────
                    if available_tabs.len() > 1 {
                        ui.add_space(2.0);
                        ui.horizontal_wrapped(|ui| {
                            ui.spacing_mut().item_spacing = Vec2::new(3.0, 3.0);
                            for (i, tab_name) in available_tabs.iter().enumerate() {
                                let is_sel  = selected_tab == i;
                                let bg      = if is_sel { palette::tab_active_bg() } else { palette::slot_bg() };
                                let fg      = if is_sel { palette::tab_active_text() } else { palette::tab_idle_text() };
                                let border  = if is_sel {
                                    Stroke::new(1.5, palette::tab_active_text())
                                } else {
                                    Stroke::new(1.0, palette::slot_border_dark())
                                };
                                let label  = egui::RichText::new(tab_name.as_str()).size(10.0).monospace().color(fg);
                                let btn    = egui::Button::new(label).fill(bg).stroke(border);
                                if ui.add(btn).clicked() {
                                    pending_tab_change = Some(i);
                                }
                            }
                        });
                        ui.add_space(4.0);
                    }

                    // ── Creative grid (filtered) ─────────────────────────────
                    ui.label(egui::RichText::new("ALL BLOCKS").size(10.0).monospace()
                        .color(palette::mode_creative()));
                    ui.add_space(3.0);

                    egui::ScrollArea::vertical().max_height(220.0).show(ui, |ui| {
                        egui::Grid::new("creative_grid")
                            .spacing(Vec2::new(SLOT_GAP, SLOT_GAP))
                            .min_col_width(SLOT_SIZE)
                            .max_col_width(SLOT_SIZE)
                            .show(ui, |ui| {
                                let mut grid_col = 0usize;
                                let creative_snap: Vec<(usize, String, [f32;3], Option<String>, Option<String>, Option<String>, u32, Option<String>)> = self.creative_items.iter().enumerate()
                                    .filter(|(_, item)| show_all_tabs || item.inventory_tab.as_deref() == Some(filter_tab_name.as_str()))
                                    .map(|(idx, item)| (idx, item.block_id.clone(), item.block_color,
                                        item.top_texture.clone(), item.side_texture.clone(),
                                        item.inventory_tab.clone(), item.amount,
                                        item.model_name.clone()))
                                    .collect();
                                for (idx, bid, color, top_nm, side_nm, _tab, amount, model_nm) in &creative_snap {
                                    let resp = ui.add_sized(
                                        [SLOT_SIZE, SLOT_SIZE],
                                        egui::Button::new("").fill(palette::slot_bg())
                                            .stroke(Stroke::new(1.5, palette::slot_border_dark())),
                                    );
                                    let top_tex  = top_nm.as_ref().and_then(|n| self.tile_textures.get(n).map(|h| h.id()));
                                    let side_tex = side_nm.as_ref().and_then(|n| self.tile_textures.get(n).map(|h| h.id()));
                                    self.draw_slot_block(ui.painter(), resp.rect, *color, top_tex, side_tex, model_nm.as_deref());

                                    if resp.clicked()                { pending_clicks.push(PendingClick::Left(SlotTarget::Creative(*idx)));  self.inventory_interacted = true; }
                                    else if resp.secondary_clicked() { pending_clicks.push(PendingClick::Right(SlotTarget::Creative(*idx))); self.inventory_interacted = true; }

                                    self.last_render_commands.push(RenderCommand {
                                        rect: resp.rect, block_id: bid.clone(), count: *amount,
                                    });

                                    grid_col += 1;
                                    if grid_col % MAIN_COLS == 0 { ui.end_row(); }
                                }
                                if grid_col % MAIN_COLS != 0 { ui.end_row(); }
                            });
                    });
                }

                // ── Hotbar row (always shown when inventory is open) ─────────
                ui.add_space(8.0);
                ui.add(egui::Separator::default().horizontal().spacing(4.0));
                ui.add_space(4.0);
                ui.label(egui::RichText::new("HOTBAR  (MMB = delete slot)")
                    .size(9.0).monospace().color(palette::section_label()));
                ui.add_space(3.0);

                {
                    let hc = self.inventory.hotbar_count();
                    let active_hi = self.inventory.active_hotbar_index;

                    // Snapshot hotbar data (releases borrow before mutable ops)
                    struct HSnap {
                        bid: String, amount: u32, color: [f32; 3],
                        top_nm: Option<String>, side_nm: Option<String>,
                        model_nm: Option<String>,
                    }
                    let hsnaps: Vec<HSnap> = (0..hc).map(|i| {
                        self.inventory.hotbar_slot(i).and_then(|s| s.item())
                            .map(|it| HSnap {
                                bid: it.block_id.clone(), amount: it.amount,
                                color: it.block_color,
                                top_nm: it.top_texture.clone(), side_nm: it.side_texture.clone(),
                                model_nm: it.model_name.clone(),
                            })
                            .unwrap_or(HSnap { bid: String::new(), amount: 0,
                                color: [0.0;3], top_nm: None, side_nm: None, model_nm: None })
                    }).collect();

                    egui::Grid::new("inv_hotbar_grid")
                        .spacing(Vec2::new(SLOT_GAP, SLOT_GAP))
                        .min_col_width(SLOT_SIZE)
                        .max_col_width(SLOT_SIZE)
                        .show(ui, |ui| {
                            for (i, snap) in hsnaps.iter().enumerate() {
                                let is_active = i == active_hi;
                                let stroke = if is_active {
                                    Stroke::new(2.0, Color32::WHITE)
                                } else {
                                    Stroke::new(1.5, palette::slot_border_dark())
                                };
                                let resp = ui.add_sized(
                                    [SLOT_SIZE, SLOT_SIZE],
                                    egui::Button::new("").fill(palette::slot_bg()).stroke(stroke),
                                );
                                if resp.clicked()                { pending_clicks.push(PendingClick::Left(SlotTarget::Hotbar(i)));   self.inventory_interacted = true; }
                                else if resp.secondary_clicked() { pending_clicks.push(PendingClick::Right(SlotTarget::Hotbar(i))); self.inventory_interacted = true; }
                                else if resp.middle_clicked()    { pending_clicks.push(PendingClick::Delete(SlotTarget::Hotbar(i))); self.inventory_interacted = true; }

                                if !snap.bid.is_empty() {
                                    let top_tex  = snap.top_nm.as_ref().and_then(|n| self.tile_textures.get(n).map(|h| h.id()));
                                    let side_tex = snap.side_nm.as_ref().and_then(|n| self.tile_textures.get(n).map(|h| h.id()));
                                    self.draw_slot_block(ui.painter(), resp.rect, snap.color, top_tex, side_tex, snap.model_nm.as_deref());
                                    if snap.amount > 1 {
                                        let p   = ui.painter();
                                        let pos = resp.rect.right_bottom() + Vec2::new(-3.0, -2.0);
                                        p.text(pos + Vec2::new(1.0,1.0), Align2::RIGHT_BOTTOM, &snap.amount.to_string(), FontId::monospace(11.0), Color32::from_black_alpha(200));
                                        p.text(pos, Align2::RIGHT_BOTTOM, &snap.amount.to_string(), FontId::monospace(11.0), palette::text_count());
                                    }
                                    if resp.hovered() {
                                        egui::show_tooltip_text(ctx, ui.layer_id(),
                                            egui::Id::new(("hb_tip", i)), snap.bid.as_str());
                                    }
                                }
                                // Slot index label
                                ui.painter().text(
                                    resp.rect.left_top() + Vec2::new(2.0, 1.0), Align2::LEFT_TOP,
                                    &(i+1).to_string(), FontId::monospace(8.0),
                                    Color32::from_white_alpha(if is_active { 180 } else { 80 }),
                                );
                                if is_active {
                                    ui.painter().rect_stroke(
                                        resp.rect.expand(2.0), Rounding::ZERO,
                                        Stroke::new(2.0, Color32::WHITE), egui::StrokeKind::Outside,
                                    );
                                }
                            }
                            ui.end_row();
                        });
                }

                // ── Trash slot ───────────────────────────────────────────────
                if self.inventory.dragged_item.is_some() {
                    ui.add_space(4.0);
                    let trash_label = egui::RichText::new("🗑 Drop here to delete")
                        .size(10.0).monospace().color(Color32::from_rgba_unmultiplied(220, 80, 80, 220));
                    let trash_resp = ui.add(
                        egui::Button::new(trash_label)
                            .fill(Color32::from_rgba_unmultiplied(80, 20, 20, 200))
                            .stroke(Stroke::new(1.5, Color32::from_rgba_unmultiplied(200, 60, 60, 255))),
                    );
                    if trash_resp.clicked() {
                        pending_clicks.push(PendingClick::Delete(SlotTarget::Trash));
                        self.inventory_interacted = true;
                    }
                }

                ui.add_space(4.0);
            });
        });

        if !open { self.inventory.is_open = false; }

        if let Some(t) = pending_tab_change {
            self.selected_tab = t;
            debug_log!("InventoryUI", "build_inventory_window",
                "Tab changed to {} ({})", t, self.available_tabs.get(t).map(|s| s.as_str()).unwrap_or("?"));
        }

        self.process_pending_clicks(pending_clicks);
    }

    // ----- Dragged item cursor ----------------------------------------------

    fn draw_dragged_item(&self, ctx: &egui::Context) {
        if let Some(ref item) = self.inventory.dragged_item {
            if let Some(ptr) = ctx.pointer_latest_pos() {
                let layer_id = egui::LayerId::new(egui::Order::Tooltip, egui::Id::new("dragged_item"));
                let painter  = ctx.layer_painter(layer_id);
                let size     = Vec2::splat(SLOT_SIZE);
                let rect     = Rect::from_center_size(ptr, size);

                painter.rect_filled(rect, Rounding::ZERO, palette::slot_active());
                painter.rect_stroke(rect, Rounding::ZERO,
                    Stroke::new(2.0, Color32::WHITE), egui::StrokeKind::Outside);

                let top_tex  = item.top_texture.as_ref().and_then(|n| self.tile_textures.get(n).map(|h| h.id()));
                let side_tex = item.side_texture.as_ref().and_then(|n| self.tile_textures.get(n).map(|h| h.id()));
                draw_block_iso(&painter, rect, item.block_color, top_tex, side_tex);

                if item.amount > 1 {
                    let pos = rect.right_bottom() + Vec2::new(-3.0, -2.0);
                    painter.text(pos + Vec2::new(1.0,1.0), Align2::RIGHT_BOTTOM,
                        &item.amount.to_string(), FontId::monospace(11.0), Color32::from_black_alpha(200));
                    painter.text(pos, Align2::RIGHT_BOTTOM,
                        &item.amount.to_string(), FontId::monospace(11.0), palette::text_count());
                }
            }
        }
    }
}

// =============================================================================
// Draw helpers
// =============================================================================

fn mul_color(c: [f32; 3], f: f32) -> [f32; 3] {
    [(c[0]*f).min(1.0), (c[1]*f).min(1.0), (c[2]*f).min(1.0)]
}

fn to_c32(c: [f32; 3]) -> Color32 {
    Color32::from_rgb(
        (c[0]*255.0).clamp(0.0,255.0) as u8,
        (c[1]*255.0).clamp(0.0,255.0) as u8,
        (c[2]*255.0).clamp(0.0,255.0) as u8,
    )
}

fn mesh_quad(mesh: &mut egui::Mesh, pts: [Pos2; 4], uvs: [[f32; 2]; 4], tint: Color32) {
    let base = mesh.vertices.len() as u32;
    for i in 0..4 {
        mesh.vertices.push(egui::epaint::Vertex {
            pos:   pts[i],
            uv:    Pos2::new(uvs[i][0], uvs[i][1]),
            color: tint,
        });
    }
    mesh.indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
}

/// Draw an isometric cube preview centred in `rect`.
/// Uses actual block face textures when `top_tex`/`side_tex` are provided.
fn draw_block_iso(
    painter: &egui::Painter,
    rect: Rect,
    color: [f32; 3],
    top_tex:  Option<egui::TextureId>,
    side_tex: Option<egui::TextureId>,
) {
    let cx = rect.center().x;
    let cy = rect.center().y;

    let sqrt3: f32 = 1.7320508;
    let slot_h = rect.height() * 0.9;
    let w  = slot_h * (sqrt3 / 4.0);
    let h  = w / sqrt3;
    let fh = 2.0 * h;

    // Shift the whole iso cube up slightly so it doesn't clip bottom
    let offset_y = -h * 0.25;
    let ax = cx;
    let ay = cy - h - fh / 2.0 + offset_y;

    // Top face diamond
    let top_pts = [
        Pos2::new(ax,     ay),
        Pos2::new(ax - w, ay + h),
        Pos2::new(ax,     ay + 2.0 * h),
        Pos2::new(ax + w, ay + h),
    ];
    // Left (west) face
    let left_pts = [
        Pos2::new(ax - w, ay + h),
        Pos2::new(ax,     ay + 2.0 * h),
        Pos2::new(ax,     ay + 2.0 * h + fh),
        Pos2::new(ax - w, ay + h + fh),
    ];
    // Right (east) face
    let right_pts = [
        Pos2::new(ax,     ay + 2.0 * h),
        Pos2::new(ax + w, ay + h),
        Pos2::new(ax + w, ay + h + fh),
        Pos2::new(ax,     ay + 2.0 * h + fh),
    ];

    // UVs: top face uses full texture mapped to diamond corners,
    // side faces use straight rectangle mapping.
    let top_uvs:   [[f32; 2]; 4] = [[0.0,0.0],[0.0,1.0],[1.0,1.0],[1.0,0.0]];
    let left_uvs:  [[f32; 2]; 4] = [[1.0,0.0],[0.0,0.0],[0.0,1.0],[1.0,1.0]];
    let right_uvs: [[f32; 2]; 4] = [[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]];

    let top_tint   = to_c32(mul_color(color, 1.25));
    let left_tint  = to_c32(mul_color(color, 0.85));
    let right_tint = to_c32(mul_color(color, 0.65));

    // Top face
    if let Some(tex) = top_tex {
        let mut m = egui::Mesh::with_texture(tex);
        mesh_quad(&mut m, top_pts, top_uvs, top_tint);
        painter.add(egui::Shape::Mesh(std::sync::Arc::new(m)));
    } else {
        painter.add(egui::Shape::convex_polygon(top_pts.to_vec(), top_tint, Stroke::NONE));
    }

    // Left face
    let ltex = side_tex.or(top_tex);
    if let Some(tex) = ltex {
        let mut m = egui::Mesh::with_texture(tex);
        mesh_quad(&mut m, left_pts, left_uvs, left_tint);
        painter.add(egui::Shape::Mesh(std::sync::Arc::new(m)));
    } else {
        painter.add(egui::Shape::convex_polygon(left_pts.to_vec(), left_tint, Stroke::NONE));
    }

    // Right face
    let rtex = side_tex.or(top_tex);
    if let Some(tex) = rtex {
        let mut m = egui::Mesh::with_texture(tex);
        mesh_quad(&mut m, right_pts, right_uvs, right_tint);
        painter.add(egui::Shape::Mesh(std::sync::Arc::new(m)));
    } else {
        painter.add(egui::Shape::convex_polygon(right_pts.to_vec(), right_tint, Stroke::NONE));
    }

    // Edge lines
    let es = Stroke::new(1.0, Color32::from_black_alpha(90));
    painter.line_segment([Pos2::new(ax, ay),     Pos2::new(ax - w, ay + h)], es);
    painter.line_segment([Pos2::new(ax, ay),     Pos2::new(ax + w, ay + h)], es);
    painter.line_segment([Pos2::new(ax - w, ay + h), Pos2::new(ax, ay + 2.0*h)], es);
    painter.line_segment([Pos2::new(ax + w, ay + h), Pos2::new(ax, ay + 2.0*h)], es);
    painter.line_segment([Pos2::new(ax, ay + 2.0*h), Pos2::new(ax, ay + 2.0*h + fh)], es);
    painter.line_segment([Pos2::new(ax - w, ay + h + fh), Pos2::new(ax, ay + 2.0*h + fh)], es);
    painter.line_segment([Pos2::new(ax + w, ay + h + fh), Pos2::new(ax, ay + 2.0*h + fh)], es);
}

fn draw_slot_bg(painter: &egui::Painter, rect: Rect, is_active: bool) {
    let bg = if is_active { palette::slot_active() } else { palette::slot_bg() };
    painter.rect_filled(rect, Rounding::ZERO, bg);
    let shadow    = Color32::from_black_alpha(150);
    let highlight = Color32::from_white_alpha(45);
    let tl = rect.left_top();  let tr = rect.right_top();
    let bl = rect.left_bottom(); let br = rect.right_bottom();
    painter.line_segment([tl, tr], Stroke::new(2.0, shadow));
    painter.line_segment([tl, bl], Stroke::new(2.0, shadow));
    painter.line_segment([bl, br], Stroke::new(2.0, highlight));
    painter.line_segment([tr, br], Stroke::new(2.0, highlight));
}

fn draw_block_iso_placeholder(painter: &egui::Painter, rect: Rect, color: [f32; 3]) {
    let cx = rect.center().x;
    let cy = rect.center().y;
    let sz = rect.height() * 0.35;
    let c  = to_c32(mul_color(color, 0.3));
    painter.rect_filled(
        Rect::from_center_size(Pos2::new(cx, cy), Vec2::splat(sz * 2.0)),
        Rounding::ZERO,
        c,
    );
}
