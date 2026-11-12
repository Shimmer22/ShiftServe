"""只读发布单个模型文件，支持 Range，供手机直接从 wins 取模型。"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', required=True, type=Path)
    parser.add_argument('--bind', default='10.29.243.183')
    parser.add_argument('--port', type=int, default=18782)
    args = parser.parse_args()
    path = args.file.resolve(strict=True)
    size = path.stat().st_size

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != '/' + path.name:
                self.send_error(404)
                return
            start, end = 0, size - 1
            header = self.headers.get('Range')
            if header:
                match = re.fullmatch(r'bytes=(\d+)-(\d*)', header)
                if not match:
                    self.send_error(416)
                    return
                start = int(match[1])
                end = min(int(match[2]) if match[2] else size - 1, size - 1)
                if not 0 <= start <= end < size:
                    self.send_error(416)
                    return
            self.send_response(206 if header else 200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Content-Length', str(end - start + 1))
            if header:
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.end_headers()
            with path.open('rb') as stream:
                self.connection.sendfile(stream, offset=start, count=end - start + 1)

    ThreadingHTTPServer((args.bind, args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
