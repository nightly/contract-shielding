"""Environment implementations and safety-layer modules."""

import sys

if __name__ == "src.environments":
    sys.modules.setdefault("environments", sys.modules[__name__])
elif __name__ == "environments":
    sys.modules.setdefault("src.environments", sys.modules[__name__])
