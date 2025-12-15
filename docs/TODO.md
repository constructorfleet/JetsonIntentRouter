# Futue Enhancement: Multi-protocol gateway support

Currently the service exposes an OpenAI-compatible API only.
Future enhancement:
- Add protocol adapters at the HTTP boundary:
  - OpenAI-compatible
  - Anthropic Messages API
  - Ollama Generate API
- Each adapter translates request/response formats
  into/from the internal OpenAI-style streaming contract.
- Router and agents remain protocol-agnostic.

# Future enhancement: Execution planning

Currently, routed clauses are executed sequentially.
Future versions may introduce:
- dependency detection between clauses
- parallel execution of independent intents
- explicit precondition handling (e.g. device power state)

Design constraints:
- routing remains deterministic
- streaming semantics remain OpenAI-compatible
- router retains control of stream termination
- agents remain unaware of execution order

```python
class RoutedClause:
    clause: str
    intent: str
    confidence: float
    agent: str

    execution:
        mode: "sequential" | "parallel"
        depends_on: list[int]  # indices of other clauses
```
