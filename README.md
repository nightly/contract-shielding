# Contract Shielding

## Code organisation

```text
.
|-- README.md
`-- src
    |-- shield
    |   |-- automaton.py      # Automata utilities used by the shielding layer.
    |   |-- contracts.py      # Contract definitions and helpers.
    |   |-- core.py           # Core shielding logic.
    |   `-- wrapper.py        # Environment wrapper for applying shields.
    |-- rl
    |   |-- icpo.py
    |   |-- ippo.py
    |   |-- ippo_contract.py
    |   |-- ippo_lagrangian.py
    |   |-- iql.py
    |   |-- iql_contract.py
    |   |-- iql_lagrangian.py
    |   |-- joint_ppo.py
    |   |-- mappo.py
    |   |-- pqn_vdn.py
    |   `-- trajectory.py     # Shared trajectory data structures.
    `-- environments
        |-- car_platoon
        |-- connector
        |-- cooking_zoo
        |-- flatland
        |-- level_based_foraging
        |-- pressure_plate
        `-- rware
```

Most environment packages follow this shape:

```text
<environment>
|-- constraints
|   |-- candidate_contracts.py
|   `-- formulas.py
|-- impl
|   `-- env.py
`-- labelled
    `-- __init__.py
```
