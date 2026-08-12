# Demo asistente juego de cartas Magic the gathering

***

# Chatbot MTG (Magic: The Gathering)

En este proyecto se ha implementado una demo funcional (PoC) de un chatbot para un call center (MTG) que ofrece a los usuarios respuestas sobre el juego de cartas **"Magic: The Gathering"**. El asistente será capaz de resolver dudas de reglas, interacciones entre cartas, búsqueda de cartas por descripción y de crear nuevas cartas usando la descripción proporcionada por el usuario. Para su implementación emplearemos la tecnología RAG, Groq paea acceder a la API de un LLM de forma gratuía, la integración del sistema con la API disponible (https://docs.magicthegathering.io/) para obtener infromación sobre las cartas existentes, y el framework para el desarrollo de agentes de IA LanGraph. 

_En pocas palabras, se trata de un sistema multiagente con un supervisor y diferentes agentes especializados en una serie de tareas que pueden realizar empleando las herramientas (tools) implementadas para ello._   

## Instalación

La instalación del sistema ñunicamente requiere la instalación de las librerías necesarias, la generación de una API Key en Groq (https://console.groq.com) para poder realizar llamadas al LLM, y en caso necesario de la mosificación de las variables de configuración del fichero MTG.py, único fichero de código necesario para la ejecución del asistente. Este programa realiza todas las tareas necesarias para el uso del asistente (gestión de la documentación, RAG, generación de agentes y herramientas, ...). El asistente no tiene una interfaz gráfica, su uso se realizará a travé del terminal de comandos (CLI), tanto en Windows como en Linux. 

En un sistema Linux, los comandos a ejecutar son los siguientes: 

```bash
pip install -r requirements.txt
export GROQ_API_KEY="tu_api_key"         
```


Mientras que en un sistema Windows ejecutaremos en el Powershell los comandos:

```bash
pip install -r requirements.txt
$env:GROQ_API_KEY = "tu_api_key"          
```
En ambos casos se debará sustituir el valor "tu_api_key" por la API Key genererada en Groq. 

## Ejecución de la demo

Para ejecutar el programa, una vez actializados los valores de las variables en el fichero MTG.py, se empleará el siguiente comando, tanto en Linux como en Windows.  

```bash
python main.py
```

Tanto en el proceso de instalación como en el de ejecición de la demo, debemos estar situados en el directorio donde se encuantran los ficheros de este repositotio. En caso contrario, se deberán emplear rutas absolutas en lugar de las rutas relativas. 

## Ejemplos

Aquí mostramos algunos ejemplos de preguntas que se pueden realizar al asistente, así como las respuestas que ofrece en cada caso:

**Ejemplo 1**

**Usuario:** ¿Qué fases hay en un turno de juego?

**Asistente MTG:** Las fases de un turno de juego en Magic: The Gathering son:

1. Fase de inicio (Beginning Phase)
2. Fase principal (Untap Step, Upkeep Step, Draw Step)
3. Fase de combate (Combat Phase)
4. Fase de fin (End Phase)

Explicación: Según el reglamento oficial de Magic: The Gathering, un turno de juego se divide en varias fases. La fase de inicio es la primera fase del turno, seguida de la fase principal, que incluye el paso de desbloqueo (Untap Step), el paso de mantenimiento (Upkeep Step) y el paso de robo (Draw Step). Luego viene la fase de combate, donde se producen los combates entre criaturas. Finalmente, el turno termina con la fase de fin.

Cita: "703. Turn-Based Actions" del reglamento oficial de Magic: The Gathering, que describe las acciones basadas en turnos que ocurren automáticamente al comienzo o fin de cada paso o fase del turno.

**Ejemplo 2**

**Usuario:** ¿Cómo funciona el mana pool?

**Asistente MTG:** El mana pool es un concepto fundamental en Magic: The Gathering que se refiere a la cantidad de mana que un jugador tiene disponible para jugar hechizos y habilidades. El mana pool se llena con mana producido por las tierras y otras fuentes de mana, y se vacía al final de cada paso y fase del turno.

Explicación: Según el reglamento oficial de Magic: The Gathering, el mana se produce mediante habilidades de mana y otros efectos, y se agrega al mana pool del jugador. El mana pool es donde se almacena el mana hasta que se utiliza para pagar costos o se vacía al final del turno. Cada jugador tiene su propio mana pool, y el mana se puede utilizar para jugar hechizos y habilidades.

Cita: "106.3. Mana is produced by the effects of mana abilities (see rule 605). It may also be produced by the effects of spells, as well as by the effects of abilities that aren’t mana abilities. A spell or ability that produces mana instructs a player to add that mana. If mana is produced by a spell, the source of that mana is that spell. If mana is produced by an ability, the source of that mana is the source of that ability (see rule 113.7)" y "106.4. When an effect instructs a player to add mana, that mana goes into a player’s mana pool. From there, it can be used to pay costs immediately, or it can stay in the player’s mana pool as unspent mana. Each player’s mana pool empties at the end of each step and phase, and the player is said to" del reglamento oficial de Magic: The Gathering.

**Ejemplo 3**

**Usuario:** Mi criatura con daño primero ha hecho daño primero, si cambio su control con un efecto antes del paso de daño regular, ¿vuelve a hacer daño?

**Asistente MTG:** Si cambias el control de tu criatura con daño primero antes del paso de daño regular, la criatura ya ha realizado su daño primero y no volverá a hacer daño en el paso de daño regular.

Explicación: Según el reglamento oficial de Magic: The Gathering, el daño primero se aplica en el paso de daño de combate, y si la criatura ha realizado su daño primero, no volverá a hacer daño en el mismo paso de daño. Si cambias el control de la criatura antes del paso de daño regular, la criatura ya ha realizado su daño primero y no estará pendiente de hacer daño en el paso de daño regular.

Cita: "509.1. First, if any attacking creatures with first strike (see rule 702.7) are dealt lethal damage, they
deal combat damage first" y "702.7. First Strike
702.7a First strike is a static ability that means ‘This creature deals combat damage first’" del reglamento oficial de Magic: The Gathering.

**Ejemplo 4**

**Usuario:** Busco una carta de color blanco de coste inferior a dos de mana que sea guerrero.

**Asistente MTG:** Lo siento, no se encontraron cartas de color blanco con un coste de maná inferior a 2 que sean del tipo Guerrero. Es posible que la carta que estás buscando no exista o que no esté en la base de datos.

Si deseas, puedo intentar buscar de nuevo con criterios ligeramente diferentes. Por ejemplo, puedo buscar cartas de color blanco con un coste de maná de 2 o menos que sean del tipo Guerrero. ¿Te gustaría que intente de nuevo?

**Ejemplo 5**

**Usuario:** ¿Cuáles son los últimos sets que han salido?

**Asistente MTG:** Los últimos sets que han salido son:

- 30th Anniversary Edition (30A) - lanzamiento: 28 de noviembre de 2022, tipo: memorabilia
- Double Masters 2022 (2X2) - lanzamiento: 8 de julio de 2022, tipo: masters

Es importante tener en cuenta que la información sobre los sets más recientes puede variar dependiendo de la fecha actual y de la fuente de la información. Si necesitas información más actualizada, te recomiendo verificar la página oficial de Magic: The Gathering o otras fuentes confiables.

**Ejemplo 6**

**Usuario:** Quiero crear una carta de Han Solo, blanca-roja, que tenga daño primero.

**Asistente MTG:** Una carta de Han Solo con daño primero sería un excelente agregado a cualquier deck. Aquí te dejo una posible implementación:

*Nombre:* Han Solo, Contrabandista

*Coste de maná:* {1}{R}{W}

*Tipo:* Criatura - Humano Piloto

*Texto de reglas:* Daño primero, vigilante

*Poder/Toughness:* 2/2

*Flavor text:* "Nunca digas nunca... a menos que estés hablando de pagar impuestos."

Justificación: El coste de maná de {1}{R}{W} refleja la naturaleza versátil de Han Solo, que puede ser jugado en decks blancos y rojos. El daño primero y la vigilancia hacen que sea una criatura formidable en el campo de batalla, capaz de dealing daño y proteger a sus aliados. El poder y la resistencia de 2/2 son adecuados para una criatura de su tamaño y habilidades.

Referencia: La carta de Han Solo se inspira en la carta "Ajani's Pridemate" {1}{W}, que tiene una habilidad similar de daño primero y vigilancia. Sin embargo, el coste de maná y las estadísticas de Han Solo están diseñadas para reflejar su personalidad y habilidades únicas en el universo de Star Wars.
