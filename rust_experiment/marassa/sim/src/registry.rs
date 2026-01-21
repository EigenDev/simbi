// =============================================================================
// registry.rs
//
// type-erased heterogeneous storage keyed by entity id.
// stores any type T keyed by (TypeId, Entity). simple and flexible.
//
// this is optional—most state lives in the typed hierarchy. the registry
// is for components that don't fit the standard structure.
//
// usage:
//   let mut reg = Registry::new();
//   reg.insert(entity, MyComponent { ... });
//   let comp: &MyComponent = reg.get(entity).unwrap();
// =============================================================================

use crate::Entity;
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;

#[derive(Default)]
pub struct Registry {
    storage: HashMap<TypeId, HashMap<Entity, Box<dyn Any + Send + Sync>>>,
}

impl Registry {
    pub fn new() -> Self {
        Self {
            storage: HashMap::new(),
        }
    }

    pub fn insert<T: Any + Send + Sync>(&mut self, entity: Entity, component: T) {
        let type_id = TypeId::of::<T>();
        self.storage
            .entry(type_id)
            .or_default()
            .insert(entity, Box::new(component));
    }

    pub fn get<T: Any + Send + Sync>(&self, entity: Entity) -> Option<&T> {
        let type_id = TypeId::of::<T>();
        self.storage.get(&type_id)?.get(&entity)?.downcast_ref()
    }

    pub fn get_mut<T: Any + Send + Sync>(&mut self, entity: Entity) -> Option<&mut T> {
        let type_id = TypeId::of::<T>();
        self.storage
            .get_mut(&type_id)?
            .get_mut(&entity)?
            .downcast_mut()
    }

    pub fn has<T: Any + Send + Sync>(&self, entity: Entity) -> bool {
        let type_id = TypeId::of::<T>();
        self.storage
            .get(&type_id)
            .map(|m| m.contains_key(&entity))
            .unwrap_or(false)
    }

    pub fn remove<T: Any + Send + Sync>(&mut self, entity: Entity) -> Option<T> {
        let type_id = TypeId::of::<T>();
        self.storage
            .get_mut(&type_id)?
            .remove(&entity)?
            .downcast()
            .ok()
            .map(|b| *b)
    }

    pub fn entities_with<T: Any + Send + Sync>(&self) -> impl Iterator<Item = Entity> + '_ {
        let type_id = TypeId::of::<T>();
        self.storage
            .get(&type_id)
            .into_iter()
            .flat_map(|m| m.keys().copied())
    }
}

impl fmt::Debug for Registry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // type-erased storage can't print values, just show structure
        let type_count = self.storage.len();
        let entity_count: usize = self.storage.values().map(|m| m.len()).sum();
        f.debug_struct("Registry")
            .field("types", &type_count)
            .field("entities", &entity_count)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct Position {
        x: f64,
        y: f64,
    }

    #[derive(Debug, PartialEq)]
    struct Velocity {
        vx: f64,
        vy: f64,
    }

    #[test]
    fn insert_and_get() {
        let mut reg = Registry::new();
        let entity = Entity::new();

        reg.insert(entity, Position { x: 1.0, y: 2.0 });

        let pos: &Position = reg.get(entity).unwrap();
        assert_eq!(pos.x, 1.0);
        assert_eq!(pos.y, 2.0);
    }

    #[test]
    fn get_mut() {
        let mut reg = Registry::new();
        let entity = Entity::new();

        reg.insert(entity, Position { x: 1.0, y: 2.0 });

        let pos: &mut Position = reg.get_mut(entity).unwrap();
        pos.x = 5.0;

        let pos: &Position = reg.get(entity).unwrap();
        assert_eq!(pos.x, 5.0);
    }

    #[test]
    fn multiple_components() {
        let mut reg = Registry::new();
        let entity = Entity::new();

        reg.insert(entity, Position { x: 1.0, y: 2.0 });
        reg.insert(entity, Velocity { vx: 3.0, vy: 4.0 });

        let pos: &Position = reg.get(entity).unwrap();
        let vel: &Velocity = reg.get(entity).unwrap();

        assert_eq!(pos.x, 1.0);
        assert_eq!(vel.vx, 3.0);
    }

    #[test]
    fn multiple_entities() {
        let mut reg = Registry::new();
        let e1 = Entity::new();
        let e2 = Entity::new();

        reg.insert(e1, Position { x: 1.0, y: 2.0 });
        reg.insert(e2, Position { x: 10.0, y: 20.0 });

        let p1: &Position = reg.get(e1).unwrap();
        let p2: &Position = reg.get(e2).unwrap();

        assert_eq!(p1.x, 1.0);
        assert_eq!(p2.x, 10.0);
    }

    #[test]
    fn has_component() {
        let mut reg = Registry::new();
        let entity = Entity::new();

        assert!(!reg.has::<Position>(entity));

        reg.insert(entity, Position { x: 1.0, y: 2.0 });

        assert!(reg.has::<Position>(entity));
        assert!(!reg.has::<Velocity>(entity));
    }

    #[test]
    fn remove_component() {
        let mut reg = Registry::new();
        let entity = Entity::new();

        reg.insert(entity, Position { x: 1.0, y: 2.0 });
        assert!(reg.has::<Position>(entity));

        let removed = reg.remove::<Position>(entity);
        assert_eq!(removed, Some(Position { x: 1.0, y: 2.0 }));
        assert!(!reg.has::<Position>(entity));
    }

    #[test]
    fn entities_with_component() {
        let mut reg = Registry::new();
        let e1 = Entity::new();
        let e2 = Entity::new();
        let e3 = Entity::new();

        reg.insert(e1, Position { x: 1.0, y: 2.0 });
        reg.insert(e2, Position { x: 3.0, y: 4.0 });
        reg.insert(e3, Velocity { vx: 5.0, vy: 6.0 });

        let with_pos: Vec<_> = reg.entities_with::<Position>().collect();
        assert_eq!(with_pos.len(), 2);
        assert!(with_pos.contains(&e1));
        assert!(with_pos.contains(&e2));
    }
}
