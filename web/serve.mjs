import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..');
const webRoot = __dirname;
const distRoot = path.join(repoRoot, 'build', 'web-release', 'web');
const modelsRoot = path.join(repoRoot, 'models');
const port = Number(process.env.PORT || 8080);

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.mjs': 'application/javascript; charset=utf-8',
  '.wasm': 'application/wasm',
  '.json': 'application/json; charset=utf-8',
  '.pb': 'application/octet-stream',
  '.bin': 'application/octet-stream',
  '.mlvc': 'application/octet-stream',
  '.css': 'text/css; charset=utf-8',
};

function resolveFile(urlPath) {
  const clean = decodeURIComponent(urlPath.split('?')[0]);
  if (clean === '/' || clean === '/index.html')
    return path.join(webRoot, 'index.html');
  if (clean.startsWith('/dist/'))
    return path.join(distRoot, clean.slice('/dist/'.length));
  if (clean.startsWith('/models/'))
    return path.join(modelsRoot, clean.slice('/models/'.length));
  return path.join(webRoot, clean.replace(/^\/+/, ''));
}

const server = http.createServer((req, res) => {
  const filePath = resolveFile(req.url || '/');
  const norm = path.normalize(filePath);
  if (!norm.startsWith(webRoot) && !norm.startsWith(distRoot) && !norm.startsWith(modelsRoot)) {
    res.writeHead(403);
    res.end('Forbidden');
    return;
  }
  fs.stat(norm, (err, st) => {
    if (err || !st.isFile()) {
      res.writeHead(404);
      res.end('Not found: ' + (req.url || ''));
      return;
    }
    const ext = path.extname(norm).toLowerCase();
    res.writeHead(200, {
      'Content-Type': MIME[ext] || 'application/octet-stream',
      'Content-Length': st.size,
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cache-Control': 'no-cache',
    });
    fs.createReadStream(norm).pipe(res);
  });
});

server.listen(port, '127.0.0.1', () => {
  console.log(`mlvc.cpp WebGPU player serving on http://127.0.0.1:${port}`);
});
