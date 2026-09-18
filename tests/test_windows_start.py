"""无需安装依赖的 Windows 启动脚本静态回归测试。"""
import re
import unittest
from pathlib import Path


START = (Path(__file__).resolve().parents[1] / 'start.bat').read_text(encoding='utf-8')
INSTALL = (Path(__file__).resolve().parents[1] / 'install.bat').read_text(encoding='utf-8')


class WindowsStartTests(unittest.TestCase):
    def test_missing_mnist_warns_in_chinese_and_continues_to_flask(self):
        missing = re.search(r'(?ims)^if not exist "data\\mnist\.npz" \(\s*(.*?)^\)', START)
        self.assertIsNotNone(missing, '启动脚本应提示缺少 MNIST 数据')
        branch = missing.group(1)
        self.assertIn('MNIST', branch)
        self.assertIn('课程页面', branch)
        self.assertIn('暂不可用', branch)
        self.assertIn('python download_data.py', branch)
        self.assertRegex(START[:missing.start()], r'(?m)^chcp 65001 >nul\s*$')
        self.assertNotRegex(branch, r'(?im)^\s*(?:exit\s*/b|pause)\b')
        self.assertRegex(START[missing.end():], r'(?m)^python server\.py\s*$')

    def test_missing_python_or_dependencies_still_exits(self):
        self.assertRegex(START, r'(?s)where python >nul 2>&1\s*if errorlevel 1 \([^)]*exit /b 1')
        self.assertRegex(START, r'(?s)python -c "import paths, flask, numpy, torch" >nul 2>&1\s*if errorlevel 1 \([^)]*exit /b 1')
    def test_successful_install_directs_to_course_before_dataset(self):
        self.assertIn('Next run: start.bat', INSTALL)
        self.assertIn('Before training: python download_data.py', INSTALL)


if __name__ == '__main__':
    unittest.main()
