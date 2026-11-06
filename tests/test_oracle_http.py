"""HTTP 正文完整即返回；连接不关闭不能延长实测 Gap，截断正文必须失败。"""
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


class OracleHttpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('c++')
        if not compiler:
            raise unittest.SkipTest('需要本机 C++ 编译器')
        cls.tmp = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.tmp.name) / 'http-test'
        source = Path(cls.tmp.name) / 'http-test.cpp'
        http = (ROOT / 'src/a2o/native/profiling/oracle.h').read_text().split('static int run_oracle')[0]
        source.write_text('#include <string>\n#include <stdexcept>\n#include <iostream>\n#include <signal.h>\n'
                          'void require(bool b,const std::string &m){if(!b)throw std::runtime_error(m);}\n' + http +
                          '\nint main(int argc,char**argv){struct sigaction a{};a.sa_handler=+[](int){};sigaction(SIGALRM,&a,nullptr);ualarm(50000,0);try{std::cout<<oracle_http(std::stoi(argv[1]),16,0,0);return 0;}'
                          'catch(const std::exception&e){std::cerr<<e.what();return 1;}}\n')
        subprocess.run([compiler, '-std=c++17', '-I' + str(ROOT / 'src/a2o/native'),
                        str(source), '-o', str(cls.binary)], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def request(self, response, keep_open=False, response_delay=0):
        with socket.socket() as server:
            server.bind(('127.0.0.1', 0)); server.listen()
            def serve():
                conn, _ = server.accept()
                with conn:
                    conn.recv(8192)
                    time.sleep(response_delay)
                    conn.sendall(response)
                    if keep_open:
                        conn.settimeout(2)
                        try:
                            conn.recv(1)  # 等客户端先关闭，模拟正文完成但 reverse 连接尚未关闭。
                        except socket.timeout:
                            pass
            thread = threading.Thread(target=serve); thread.start()
            try:
                return subprocess.run([str(self.binary), str(server.getsockname()[1])],
                                      capture_output=True, timeout=1)
            finally:
                thread.join(timeout=3)

    def test_complete_body_does_not_wait_for_connection_close(self):
        result = self.request(b'HTTP/1.0 200 OK\r\nContent-Length: 4\r\n\r\nnote', keep_open=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, b'note')

    def test_truncated_body_is_rejected(self):
        result = self.request(b'HTTP/1.0 200 OK\r\nContent-Length: 9\r\n\r\nnote')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'HTTP incomplete response', result.stderr)

    def test_interrupted_read_continues_same_request(self):
        result = self.request(b'HTTP/1.0 200 OK\r\nContent-Length: 4\r\n\r\nnote', response_delay=.15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, b'note')

    def test_error_status_is_rejected(self):
        result = self.request(b'HTTP/1.0 500 Error\r\nContent-Length: 4\r\n\r\nnote')
        self.assertNotEqual(result.returncode, 0)

    def test_missing_length_is_rejected(self):
        result = self.request(b'HTTP/1.0 200 OK\r\nX: 1\r\n\r\nnote')
        self.assertNotEqual(result.returncode, 0)
