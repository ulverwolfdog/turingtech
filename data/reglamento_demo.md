# Reglas de Magic: The Gathering — Extracto de referencia (DEMO)

> **NOTA IMPORTANTE:** Este fichero es un extracto reducido y simplificado
> creado únicamente para poder probar el pipeline de RAG en la demo. El
> cliente debe sustituirlo por el **reglamento oficial completo (Comprehensive
> Rules)** en PDF antes de pasar a producción. Ver `README.md` sección
> "Sustituir el reglamento por el oficial".

## 1. Estructura de un turno

Un turno de Magic se divide en las siguientes fases y pasos, en este orden:

1. **Fase de inicio (Beginning Phase)**
   - Paso de enderezar (Untap step): el jugador activo endereza sus permanentes.
     No se usa la pila y ningún jugador recibe prioridad.
   - Paso de mantenimiento (Upkeep step): se resuelven disparadores de "al
     comienzo del mantenimiento". Los jugadores reciben prioridad.
   - Paso de robo (Draw step): el jugador activo roba una carta. Los
     jugadores reciben prioridad después del robo.

2. **Fase principal (Main Phase 1)**: el jugador activo puede jugar cartas
   de tierra, lanzar hechizos y activar habilidades. Es la única fase (junto
   con la segunda fase principal) en la que se pueden jugar tierras y lanzar
   hechizos que no sean de instantáneo, siempre que la pila esté vacía.

3. **Fase de combate (Combat Phase)**
   - Comienzo de combate (Beginning of Combat step).
   - Declarar atacantes (Declare Attackers step): el jugador activo declara
     qué criaturas atacan.
   - Declarar bloqueadores (Declare Blockers step): el jugador defensor
     declara qué criaturas bloquean, y a qué atacante.
   - Daño de combate (Combat Damage step): se asigna y se inflige el daño
     de combate simultáneamente (salvo criaturas con daño primero o daño
     doble, ver sección 4).
   - Fin de combate (End of Combat step).

4. **Segunda fase principal (Main Phase 2)**: igual que la Main Phase 1.

5. **Fase final (Ending Phase)**
   - Paso final (End step): se resuelven disparadores de "al comienzo del
     paso final".
   - Paso de limpieza (Cleanup step): el jugador activo descarta hasta su
     máximo de mano (normalmente 7 cartas); el daño marcado se elimina de
     las permanentes y terminan los efectos de "hasta fin de turno". Normalmente
     no se otorga prioridad, salvo que ocurra algún disparador.

## 2. El maná y el mana pool

- El maná es el recurso que se gasta para jugar cartas y activar
  habilidades. Hay cinco colores de maná (blanco, azul, negro, rojo, verde)
  más el maná incoloro.
- El maná se genera normalmente activando la habilidad de maná de una
  tierra (p. ej. "T: Añade {R}."), aunque también hay criaturas, artefactos
  y hechizos que producen maná.
- Todo el maná no gastado se vacía del "mana pool" al final de cada paso y
  fase (regla de vaciado del maná). Por tanto, el maná generado en un paso
  no puede usarse en el siguiente si no se gasta antes.
- El coste de maná de un hechizo se paga como parte del proceso de lanzarlo,
  antes de que se añada a la pila y se resuelva.

## 3. La pila (the Stack) y prioridad

- Cuando un jugador lanza un hechizo o activa una habilidad, este se coloca
  en la pila. Los hechizos y habilidades en la pila se resuelven en orden
  "último en entrar, primero en salir" (LIFO).
- Los jugadores reciben prioridad para responder antes de que se resuelva
  cualquier objeto de la pila. Solo cuando todos los jugadores pasan de
  forma consecutiva sin realizar más acciones, el objeto superior de la
  pila se resuelve.

## 4. Daño de combate: daño primero y daño doble

- **Daño primero (First Strike):** una criatura con esta habilidad asigna y
  causa daño de combate en un paso adicional de daño de combate, ANTES que
  las criaturas sin daño primero. Si en ese primer paso una criatura
  bloqueadora o atacante sin daño primero es destruida, no llega a asignar
  ni causar daño en el segundo paso (porque ya no está en el campo de
  batalla).
- **Daño doble (Double Strike):** una criatura con esta habilidad asigna y
  causa daño de combate TANTO en el paso de daño primero COMO en el paso de
  daño regular. Es decir, causa daño dos veces si sigue en el campo de
  batalla en ambos pasos.
- Si ninguna criatura en combate tiene daño primero ni daño doble, solo
  existe un único paso de daño de combate (el regular), y todas las
  criaturas asignan y causan daño simultáneamente en ese paso.
- Una habilidad de daño primero o daño doble solo tiene efecto en el
  combate en el que la criatura la tiene en el momento en que empieza el
  paso de daño correspondiente. Si una criatura pierde la habilidad de daño
  primero después del primer paso de daño de combate pero antes del
  segundo (por ejemplo, por un efecto que se la quita, o porque cambia de
  controlador), esa criatura NO vuelve a causar daño en el paso regular si
  ya causó daño en el paso de daño primero, salvo que también tenga daño
  doble en ese momento. En términos generales: lo relevante es qué
  habilidades tiene la criatura en el momento exacto en que se determina
  qué criaturas participan en cada paso de daño, no las que tenía cuando
  empezó el combate.

## 5. Cambio de controlador de una criatura durante el combate

- Si una criatura atacante o bloqueadora cambia de controlador después de
  haber sido declarada como atacante/bloqueadora, sigue atacando o
  bloqueando igualmente (conserva su estado de combate), pero ahora
  pertenece a las decisiones del nuevo controlador para el resto del
  combate (por ejemplo, para asignar daño si tiene la habilidad de
  arrollar).
- El cambio de controlador no elimina el daño que la criatura ya haya
  causado o recibido previamente en el combate; el daño marcado permanece
  hasta el paso de limpieza.

## 6. Acciones basadas en estado

- Las acciones basadas en estado (state-based actions) se comprueban
  constantemente, sin usar la pila. Ejemplos relevantes: una criatura con
  daño marcado igual o superior a su resistencia es destruida; una
  criatura con resistencia 0 o menos va al cementerio; un jugador con 0 o
  menos vidas pierde la partida.
