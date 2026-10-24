import time
import heapq
from collections import deque
from typing import List, Tuple, Dict
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="Bomberman Full AI Game Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
# 1. MODEL DATA
# --------------------------------------------------------------------------
class MultiPathRequest(BaseModel):
    grid: List[List[int]]
    enemies: List[List[int]]
    target: List[int]
    algorithm: str

# --------------------------------------------------------------------------
# 2. LOGIKA ALGORITMA PATHFINDING
# --------------------------------------------------------------------------
def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def get_neighbors(cell: Tuple[int, int], grid: List[List[int]]) -> List[Tuple[Tuple[int, int], int]]:
    rows, cols = len(grid), len(grid[0])
    r, c = cell
    neighbors = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            val = grid[nr][nc]
            if val == 0:
                neighbors.append(((nr, nc), 1))
            elif val == 2:
                neighbors.append(((nr, nc), 6))
    return neighbors

def reconstruct_path(came_from: Dict, start: Tuple[int, int], target: Tuple[int, int]) -> List[List[int]]:
    path = []
    curr = target
    while curr in came_from:
        path.append([curr[0], curr[1]])
        curr = came_from[curr]
    path.append([start[0], start[1]])
    path.reverse()
    return path

def run_search(algo: str, grid: List[List[int]], start: Tuple[int, int], target: Tuple[int, int]):
    if start == target:
        return [[start[0], start[1]]], 0

    nodes_explored = 0
    came_from = {}

    if algo == "BFS":
        queue = deque([start])
        visited = {start}
        while queue:
            curr = queue.popleft()
            nodes_explored += 1
            if curr == target:
                return reconstruct_path(came_from, start, target), nodes_explored
            for (nxt, _) in get_neighbors(curr, grid):
                if nxt not in visited:
                    visited.add(nxt)
                    came_from[nxt] = curr
                    queue.append(nxt)

    elif algo == "UCS":
        pq = [(0, start)]
        g_score = {start: 0}
        while pq:
            curr_g, curr = heapq.heappop(pq)
            nodes_explored += 1
            if curr == target:
                return reconstruct_path(came_from, start, target), nodes_explored
            if curr_g > g_score.get(curr, float('inf')):
                continue
            for (nxt, cost) in get_neighbors(curr, grid):
                new_g = curr_g + cost
                if new_g < g_score.get(nxt, float('inf')):
                    g_score[nxt] = new_g
                    came_from[nxt] = curr
                    heapq.heappush(pq, (new_g, nxt))

    elif algo == "GBFS":
        pq = [(manhattan_distance(start, target), start)]
        visited = {start}
        while pq:
            _, curr = heapq.heappop(pq)
            nodes_explored += 1
            if curr == target:
                return reconstruct_path(came_from, start, target), nodes_explored
            for (nxt, _) in get_neighbors(curr, grid):
                if nxt not in visited:
                    visited.add(nxt)
                    came_from[nxt] = curr
                    heapq.heappush(pq, (manhattan_distance(nxt, target), nxt))

    elif algo == "A_STAR":
        pq = [(manhattan_distance(start, target), 0, start)]
        g_score = {start: 0}
        while pq:
            f, curr_g, curr = heapq.heappop(pq)
            nodes_explored += 1
            if curr == target:
                return reconstruct_path(came_from, start, target), nodes_explored
            if curr_g > g_score.get(curr, float('inf')):
                continue
            for (nxt, cost) in get_neighbors(curr, grid):
                new_g = curr_g + cost
                if new_g < g_score.get(nxt, float('inf')):
                    g_score[nxt] = new_g
                    came_from[nxt] = curr
                    new_f = new_g + manhattan_distance(nxt, target)
                    heapq.heappush(pq, (new_f, new_g, nxt))

    return [], nodes_explored

# --------------------------------------------------------------------------
# 3. ENDPOINT REST API
# --------------------------------------------------------------------------
@app.post("/api/find-paths-multi")
def find_paths_multi(req: MultiPathRequest):
    start_time = time.perf_counter()
    target_tuple = (req.target[0], req.target[1])
    
    results = []
    total_nodes = 0

    for enemy_pos in req.enemies:
        start_tuple = (enemy_pos[0], enemy_pos[1])
        path, nodes = run_search(req.algorithm.upper(), req.grid, start_tuple, target_tuple)
        total_nodes += nodes
        next_move = path[1] if len(path) > 1 else (path[0] if path else enemy_pos)
        results.append({
            "enemy_pos": enemy_pos,
            "next_move": next_move,
            "full_path": path
        })

    execution_time_ms = round((time.perf_counter() - start_time) * 1000, 3)

    return {
        "status": "success",
        "algorithm": req.algorithm,
        "results": results,
        "metrics": {
            "total_nodes_explored": total_nodes,
            "execution_time_ms": execution_time_ms
        }
    }

# --------------------------------------------------------------------------
# 4. FRONTEND GUI - FIXED CANVAS & BOT LOGIC
# --------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def get_frontend():
    return """
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <title>Bomberman AI - Fixed Simulation</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, sans-serif; background: #181825; color: #cdd6f4; text-align: center; margin: 15px; }
            #container { display: flex; justify-content: center; gap: 20px; margin-top: 10px; }
            canvas { border: 3px solid #cba6f7; background: #1e1e2e; border-radius: 8px; }
            #panel { background: #313244; padding: 20px; border-radius: 8px; width: 300px; text-align: left; }
            select, button { padding: 10px; font-size: 14px; width: 100%; margin-top: 10px; border-radius: 4px; border: none; cursor: pointer; }
            button { background: #a6e3a1; color: #11111b; font-weight: bold; }
            button:hover { background: #94e2d5; }
            .stat { margin-top: 10px; font-weight: bold; background: #45475a; padding: 8px; border-radius: 4px; }
            .key-legend { margin-top: 15px; font-size: 12px; line-height: 1.6; color: #bac2de; }
            #statusMsg { font-size: 20px; font-weight: bold; height: 30px; margin-top: 10px; }
            .reload-bar-bg { background: #313244; border: 1px solid #cba6f7; height: 12px; border-radius: 6px; margin-top: 5px; overflow: hidden; }
            .reload-bar-fill { background: #f9e2af; height: 100%; width: 100%; transition: width 0.1s linear; }
        </style>
    </head>
    <body>
        <h2>Bomberman AI - Fixed Simulation</h2>
        <div id="statusMsg"></div>

        <div id="container">
            <canvas id="gameCanvas" width="520" height="440"></canvas>

            <div id="panel">
                <label><b>Pilih Algoritma AI Musuh:</b></label>
                <select id="algoSelect">
                    <option value="A_STAR" selected>A* Search Algorithm</option>
                    <option value="GBFS">Greedy Best-First Search (GBFS)</option>
                    <option value="BFS">Breadth-First Search (BFS)</option>
                    <option value="UCS">Uniform Cost Search (UCS)</option>
                </select>

                <button id="restartBtn">🔄 Restart Game</button>

                <div class="stat">Nyawa Player: <span id="healthVal">❤️ ❤️ ❤️</span></div>
                <div class="stat">
                    Amunisi Bom: <span id="ammoVal">💣 💣 💣</span>
                    <div class="reload-bar-bg"><div id="reloadFill" class="reload-bar-fill"></div></div>
                </div>
                <div class="stat">Musuh Tersisa: <span id="enemyCount">1</span></div>
                <div class="stat">Total Node Visited: <span id="nodesVal">0</span></div>
                <div class="stat">Waktu Exec API: <span id="timeVal">0 ms</span></div>

                <div class="key-legend">
                    <b>Kontrol Game:</b><br>
                    🏃 <b>WASD / Panah</b>: Bergerak<br>
                    💣 <b>Spasi</b>: Pasang Bom (Maks 3 & Cooldown 3s)<br><br>
                    <b>Fitur AI:</b> Bot aktif memasang bom untuk menghancurkan dinding & menyerang pemain jika berada dalam jangkauan.
                </div>
            </div>
        </div>

        <script>
            const canvas = document.getElementById('gameCanvas');
            const ctx = canvas.getContext('2d');
            const CELL = 40;
            const ROWS = 11, COLS = 13;

            const MAP_TEMPLATE = [
                [1,1,1,1,1,1,1,1,1,1,1,1,1],
                [1,0,0,2,2,2,2,2,2,2,0,0,1],
                [1,0,1,2,1,2,1,2,1,2,1,0,1],
                [1,2,2,2,2,2,2,2,2,2,2,2,1],
                [1,2,1,2,1,2,1,2,1,2,1,2,1],
                [1,2,2,2,2,0,2,2,2,2,2,2,1],
                [1,2,1,2,1,2,1,2,1,2,1,2,1],
                [1,2,2,2,2,2,2,2,2,2,2,2,1],
                [1,0,1,2,1,2,1,2,1,2,1,0,1],
                [1,0,0,2,2,2,2,2,2,2,0,0,1],
                [1,1,1,1,1,1,1,1,1,1,1,1,1]
            ];

            let grid, player, enemies, bombs, explosions;
            let playerPixel, enemyPixels;
            let gameOver = false;
            let aiPaths = [];
            let isFetching = false;

            function initGame() {
                grid = JSON.parse(JSON.stringify(MAP_TEMPLATE));
                
                player = { 
                    r: 1, c: 1, 
                    alive: true, 
                    health: 3, 
                    iFrames: 0,
                    ammo: 3,
                    maxAmmo: 3,
                    isReloading: false,
                    reloadTimer: 0,
                    reloadMaxTimer: 180
                };

                enemies = [
                    { id: 1, r: 9, c: 11, alive: true }
                ];
                
                bombs = [];
                explosions = [];
                gameOver = false;
                isFetching = false;
                document.getElementById('statusMsg').innerText = '';
                updateHealthDisplay();
                updateAmmoDisplay();

                playerPixel = { x: player.c * CELL, y: player.r * CELL };
                enemyPixels = enemies.map(e => ({ x: e.c * CELL, y: e.r * CELL }));
            }

            function updateHealthDisplay() {
                let hearts = "";
                for (let i = 0; i < player.health; i++) hearts += "❤️ ";
                if (player.health <= 0) hearts = "💀 Mati";
                document.getElementById('healthVal').innerText = hearts;
            }

            function updateAmmoDisplay() {
                const ammoEl = document.getElementById('ammoVal');
                const fillEl = document.getElementById('reloadFill');

                if (player.isReloading) {
                    let progress = ((player.reloadMaxTimer - player.reloadTimer) / player.reloadMaxTimer) * 100;
                    fillEl.style.width = `${progress}%`;
                    ammoEl.innerText = `🔄 Reloading...`;
                    ammoEl.style.color = "#f9e2af";
                } else {
                    fillEl.style.width = `100%`;
                    let icons = "";
                    for (let i = 0; i < player.ammo; i++) icons += "💣 ";
                    ammoEl.innerText = icons || "Kosong";
                    ammoEl.style.color = "#cdd6f4";
                }
            }

            function damagePlayer() {
                if (player.iFrames <= 0 && player.alive) {
                    player.health--;
                    player.iFrames = 60;
                    updateHealthDisplay();

                    if (player.health <= 0) {
                        player.alive = false;
                        setGameOver(false);
                    }
                }
            }

            function getDangerZone() {
                let danger = new Set();
                const dirs = [[0,0], [-1,0], [1,0], [0,-1], [0,1]];
                bombs.forEach(b => {
                    dirs.forEach(([dr, dc]) => {
                        let nr = b.r + dr, nc = b.c + dc;
                        if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS && grid[nr][nc] !== 1) {
                            danger.add(`${nr},${nc}`);
                        }
                    });
                });
                return danger;
            }

            function getEscapeMove(startR, startC, dangerZone) {
                if (!dangerZone.has(`${startR},${startC}`)) {
                    return { nextStep: [startR, startC], safeFound: true };
                }

                let queue = [[startR, startC, []]];
                let visited = new Set([`${startR},${startC}`]);
                const dirs = [[-1,0], [1,0], [0,-1], [0,1]];

                while (queue.length > 0) {
                    let [r, c, path] = queue.shift();

                    if (!dangerZone.has(`${r},${c}`)) {
                        return {
                            nextStep: path.length > 0 ? path[0] : [r, c],
                            safeFound: true
                        };
                    }

                    for (let [dr, dc] of dirs) {
                        let nr = r + dr, nc = c + dc;
                        let key = `${nr},${nc}`;
                        if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS) {
                            if (grid[nr][nc] === 0 && !visited.has(key)) {
                                visited.add(key);
                                queue.push([nr, nc, [...path, [nr, nc]]]);
                            }
                        }
                    }
                }
                return { nextStep: [startR, startC], safeFound: false };
            }

            async function moveEnemiesRealtime() {
                if (gameOver || !player.alive || isFetching) return;
                const aliveEnemies = enemies.filter(e => e.alive);
                if (aliveEnemies.length === 0) return;

                isFetching = true;
                const algo = document.getElementById('algoSelect').value;
                const dangerZone = getDangerZone();

                try {
                    const res = await fetch('/api/find-paths-multi', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            grid: grid,
                            enemies: aliveEnemies.map(e => [e.r, e.c]),
                            target: [player.r, player.c],
                            algorithm: algo
                        })
                    });
                    const data = await res.json();
                    if (data.status === 'success') {
                        document.getElementById('nodesVal').innerText = data.metrics.total_nodes_explored;
                        document.getElementById('timeVal').innerText = data.metrics.execution_time_ms + ' ms';
                        
                        aiPaths = data.results.map(r => r.full_path);

                        data.results.forEach((resItem, idx) => {
                            const enemyObj = aliveEnemies[idx];
                            if (!enemyObj) return;

                            if (dangerZone.has(`${enemyObj.r},${enemyObj.c}`)) {
                                let escapeResult = getEscapeMove(enemyObj.r, enemyObj.c, dangerZone);
                                enemyObj.r = escapeResult.nextStep[0];
                                enemyObj.c = escapeResult.nextStep[1];
                                return;
                            }

                            const distToPlayer = Math.abs(enemyObj.r - player.r) + Math.abs(enemyObj.c - player.c);
                            const inLineOfSight = (enemyObj.r === player.r && Math.abs(enemyObj.c - player.c) <= 2) ||
                                                  (enemyObj.c === player.c && Math.abs(enemyObj.r - player.r) <= 2);
                            const wantOffensiveBomb = (distToPlayer <= 2 && inLineOfSight);

                            if (resItem.next_move.length > 0) {
                                const [nr, nc] = resItem.next_move;
                                
                                if (grid[nr][nc] === 2 || wantOffensiveBomb) {
                                    let tempDangerZone = new Set(dangerZone);
                                    const dirs = [[0,0], [-1,0], [1,0], [0,-1], [0,1]];
                                    dirs.forEach(([dr, dc]) => {
                                        let tr = enemyObj.r + dr, tc = enemyObj.c + dc;
                                        if (tr >= 0 && tr < ROWS && tc >= 0 && tc < COLS && grid[tr][tc] !== 1) {
                                            tempDangerZone.add(`${tr},${tc}`);
                                        }
                                    });

                                    let escapeResult = getEscapeMove(enemyObj.r, enemyObj.c, tempDangerZone);

                                    if (escapeResult.safeFound) {
                                        placeBotBomb(enemyObj.r, enemyObj.c);
                                        enemyObj.r = escapeResult.nextStep[0];
                                        enemyObj.c = escapeResult.nextStep[1];
                                        return;
                                    }
                                }

                                if (grid[nr][nc] === 0 && !dangerZone.has(`${nr},${nc}`)) {
                                    enemyObj.r = nr;
                                    enemyObj.c = nc;
                                }
                            }
                        });
                    }
                } catch(e) {
                    console.error("Error API:", e);
                } finally {
                    isFetching = false;
                }
                document.getElementById('enemyCount').innerText = enemies.filter(e => e.alive).length;
            }

            function placeBomb() {
                if (!player.alive || gameOver || player.isReloading || player.ammo <= 0) return;
                
                if (!bombs.some(b => b.r === player.r && b.c === player.c)) {
                    bombs.push({ r: player.r, c: player.c, timer: 120, maxTimer: 120 });
                    grid[player.r][player.c] = 3;
                    
                    player.ammo--;
                    updateAmmoDisplay();

                    if (player.ammo === 0) {
                        player.isReloading = true;
                        player.reloadTimer = player.reloadMaxTimer;
                    }
                }
            }

            function placeBotBomb(r, c) {
                if (!bombs.some(b => b.r === r && b.c === c)) {
                    bombs.push({ r: r, c: c, timer: 100, maxTimer: 100 });
                    grid[r][c] = 3;
                }
            }

            function updateBombsAndExplosions() {
                for (let i = bombs.length - 1; i >= 0; i--) {
                    let b = bombs[i];
                    b.timer--;
                    if (b.timer <= 0) {
                        grid[b.r][b.c] = 0;
                        let bombR = b.r, bombC = b.c;
                        bombs.splice(i, 1);
                        explodeBomb(bombR, bombC);
                    }
                }

                for (let i = explosions.length - 1; i >= 0; i--) {
                    let ex = explosions[i];
                    ex.timer--;
                    if (ex.timer <= 0) explosions.splice(i, 1);
                }

                if (player.iFrames > 0) player.iFrames--;

                if (player.isReloading) {
                    player.reloadTimer--;
                    updateAmmoDisplay();
                    if (player.reloadTimer <= 0) {
                        player.isReloading = false;
                        player.ammo = player.maxAmmo;
                        updateAmmoDisplay();
                    }
                }
            }

            function explodeBomb(r, c) {
                const dirs = [[0,0], [-1,0], [1,0], [0,-1], [0,1]];

                dirs.forEach(([dr, dc]) => {
                    let nr = r + dr, nc = c + dc;
                    if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS) {
                        if (grid[nr][nc] === 1) return;

                        explosions.push({ r: nr, c: nc, timer: 20 });

                        if (grid[nr][nc] === 2) {
                            grid[nr][nc] = 0;
                        }

                        if (grid[nr][nc] === 3) {
                            let chainedBomb = bombs.find(b => b.r === nr && b.c === nc);
                            if (chainedBomb) {
                                chainedBomb.timer = 0;
                            }
                        }

                        if (player.r === nr && player.c === nc) {
                            damagePlayer();
                        }
                        enemies.forEach(e => {
                            if (e.alive && e.r === nr && e.c === nc) e.alive = false;
                        });
                    }
                });

                if (enemies.every(e => !e.alive) && player.alive) setGameOver(true);
            }

            function setGameOver(isWin) {
                gameOver = true;
                const msg = document.getElementById('statusMsg');
                if (isWin) {
                    msg.innerText = "🎉 KAMU MENANG! Bot berhasil dikalahkan!";
                    msg.style.color = "#a6e3a1";
                } else {
                    msg.innerText = "💀 GAME OVER! Nyawa kamu habis terkena ledakan!";
                    msg.style.color = "#f38ba8";
                }
            }

            function lerp(start, end, amt) { return (1 - amt) * start + amt * end; }

            function drawPlayer(x, y) {
                if (player.iFrames % 10 > 5) return;

                ctx.fillStyle = '#50fa7b';
                ctx.beginPath();
                ctx.arc(x + CELL/2, y + CELL/2, CELL/2 - 4, 0, Math.PI*2);
                ctx.fill();
                ctx.fillStyle = '#1e1e2e';
                ctx.fillRect(x + CELL/2 - 7, y + CELL/2 - 5, 4, 6);
                ctx.fillRect(x + CELL/2 + 3, y + CELL/2 - 5, 4, 6);
            }

            function drawEnemyBot(x, y) {
                ctx.fillStyle = '#bd93f9';
                ctx.fillRect(x + 6, y + 8, CELL - 12, CELL - 12);
                ctx.fillStyle = '#ff79c6';
                ctx.fillRect(x + CELL/2 - 2, y + 2, 4, 6);
                ctx.beginPath();
                ctx.arc(x + CELL/2, y + 2, 3, 0, Math.PI*2);
                ctx.fill();
                ctx.fillStyle = '#ff5555';
                ctx.fillRect(x + 10, y + 14, 6, 6);
                ctx.fillRect(x + CELL - 16, y + 14, 6, 6);
            }

            function drawBomb(x, y, timer, maxTimer) {
                let pulse = (timer < 30 && timer % 6 < 3) ? 2 : 0;

                ctx.fillStyle = pulse ? '#ff5555' : '#21222c';
                ctx.beginPath();
                ctx.arc(x + CELL/2, y + CELL/2 + 2, CELL/2 - 6 + pulse, 0, Math.PI*2);
                ctx.fill();

                ctx.fillStyle = '#6272a4';
                ctx.beginPath();
                ctx.arc(x + CELL/2 - 4, y + CELL/2 - 2, 3, 0, Math.PI*2);
                ctx.fill();

                // PERBAIKAN SINTAKS: ctx.moveTo(x, y)
                ctx.strokeStyle = '#ffb86c';
                ctx.lineWidth = 3;
                ctx.beginPath();
                ctx.moveTo(x + CELL/2, y + 10);
                ctx.lineTo(x + CELL/2 + 6, y + 4);
                ctx.stroke();

                ctx.fillStyle = pulse ? '#f9e2af' : '#ff5555';
                ctx.beginPath();
                ctx.arc(x + CELL/2 + 6, y + 4, 4 + (pulse ? 2 : 0), 0, Math.PI*2);
                ctx.fill();
            }

            function render() {
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                for (let r = 0; r < ROWS; r++) {
                    for (let c = 0; c < COLS; c++) {
                        if (grid[r][c] === 1) ctx.fillStyle = '#45475a';
                        else if (grid[r][c] === 2) ctx.fillStyle = '#ffb86c';
                        else ctx.fillStyle = '#1e1e2e';
                        ctx.fillRect(c * CELL, r * CELL, CELL - 1, CELL - 1);
                    }
                }

                ctx.fillStyle = 'rgba(189, 147, 249, 0.25)';
                aiPaths.forEach(path => {
                    path.forEach(p => {
                        ctx.fillRect(p[1] * CELL + 10, p[0] * CELL + 10, CELL - 20, CELL - 20);
                    });
                });

                bombs.forEach(b => drawBomb(b.c * CELL, b.r * CELL, b.timer, b.maxTimer));

                ctx.fillStyle = 'rgba(255, 85, 85, 0.85)';
                explosions.forEach(ex => ctx.fillRect(ex.c * CELL, ex.r * CELL, CELL - 1, CELL - 1));

                if (player.alive) {
                    playerPixel.x = lerp(playerPixel.x, player.c * CELL, 0.25);
                    playerPixel.y = lerp(playerPixel.y, player.r * CELL, 0.25);
                    drawPlayer(playerPixel.x, playerPixel.y);
                }

                enemies.forEach((e, idx) => {
                    if (e.alive) {
                        enemyPixels[idx].x = lerp(enemyPixels[idx].x, e.c * CELL, 0.15);
                        enemyPixels[idx].y = lerp(enemyPixels[idx].y, e.r * CELL, 0.15);
                        drawEnemyBot(enemyPixels[idx].x, enemyPixels[idx].y);
                    }
                });
            }

            let frameCount = 0;
            function gameLoop() {
                frameCount++;
                updateBombsAndExplosions();
                render();

                if (frameCount % 30 === 0) {
                    moveEnemiesRealtime();
                }

                requestAnimationFrame(gameLoop);
            }

            window.addEventListener('keydown', (e) => {
                if (gameOver || !player.alive) return;

                let nr = player.r, nc = player.c;
                if (e.key === 'w' || e.key === 'ArrowUp') nr--;
                if (e.key === 's' || e.key === 'ArrowDown') nr++;
                if (e.key === 'a' || e.key === 'ArrowLeft') nc--;
                if (e.key === 'd' || e.key === 'ArrowRight') nc++;

                if (grid[nr] && grid[nr][nc] === 0) {
                    player.r = nr;
                    player.c = nc;
                }

                if (e.code === 'Space') {
                    e.preventDefault();
                    placeBomb();
                }
            });

            document.getElementById('algoSelect').addEventListener('change', moveEnemiesRealtime);
            document.getElementById('restartBtn').addEventListener('click', initGame);

            initGame();
            gameLoop();
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)