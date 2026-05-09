"""
Lab1 入口：运行测试或数据生成。

  python main.py           # 运行 pytest
"""

import sys

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main(["-v", "tests/"]))
