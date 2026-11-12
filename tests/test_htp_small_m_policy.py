"""在主机编译真实DSP路由谓词，验证白名单边界和FFN Down负对照。"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SmallMPolicyTest(unittest.TestCase):
    def test_policy_boundaries(self):
        compiler = shutil.which('cc')
        if not compiler:
            self.skipTest('C compiler unavailable')
        source = r'''
#include <assert.h>
#include "backend/htp_small_m_policy.h"
int main(void) {
    for (int m=1;m<=128;++m) {
        assert(a2o_htp_small_m_allowed(m,4096,8192,1,0) == (A2O_HTP_SMALL_M_POLICY!=0 && m==31));
        assert(a2o_htp_small_m_allowed(m,4096,12288,1,0) == (A2O_HTP_SMALL_M_POLICY!=0 && m==31));
        assert(a2o_htp_small_m_allowed(m,12288,4096,1,0) == (A2O_HTP_SMALL_M_POLICY==1 && m==31));
        assert(!a2o_htp_small_m_allowed(m,4096,8192,0,0));
        assert(!a2o_htp_small_m_allowed(m,4096,8192,1,1));
    }
    assert(a2o_htp_small_m_allowed(31,2048,4096,1,0) == (A2O_HTP_SMALL_M_POLICY==1));
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            path = work / 'test.c'
            path.write_text(source)
            for policy in range(3):
                binary = work / str(policy)
                subprocess.run([compiler, '-Wall', '-Werror', '-std=c11',
                    '-DA2O_HTP_SMALL_M_POLICY=' + str(policy), '-I' + str(ROOT / 'src/a2o/native'),
                    str(path), '-o', str(binary)], check=True, capture_output=True)
                subprocess.run([str(binary)], check=True)
