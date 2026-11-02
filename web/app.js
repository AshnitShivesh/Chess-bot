/**
 * PIXEL CHESS - Retro Arcade Frontend Controller & AI Connector
 * 
 * Features:
 * - 100% self-contained 16x16 pixel-art vector piece graphics (zero external assets)
 * - 8-bit Web Audio API sound synthesizer (blips, crunches, victory arpeggios)
 * - Full legal chess state machine (castling, en passant, promotion, pins, checkmate)
 * - Zero-config HTTP connector to Python backend (server.py -> champion.pt + search.py)
 * - Automatic offline fallback AI (minimax + piece-square tables) if Python is offline
 * - Interactive board, move log terminal, real-time eval bar, and telemetry HUD
 */

(() => {
  'use strict';

  // ==========================================================================
  // 1. 16x16 PIXEL ART PIECE ASSETS
  // ==========================================================================
  // Symbols: . = transparent, B = border, M = main, H = highlight, S = shadow
  const PIXEL_PATTERNS = {
    P: [
      "................",
      "................",
      "......BBBB......",
      ".....BMMMMB.....",
      ".....BMHHMB.....",
      ".....BMMMMB.....",
      "......BBBB......",
      "......BMMB......",
      ".....BMMMMB.....",
      "....BMMMMMMB....",
      "...BMMMMMMMMB...",
      "..BMMMSSSSMMMB..",
      "..BBBBBBBBBBBB..",
      ".BMMMMMMMMMMMMB.",
      ".BBBBBBBBBBBBBB.",
      "................"
    ],
    R: [
      "................",
      "................",
      "...BB..BB..BB...",
      "...BM..BM..MB...",
      "...BMMMMMMMMB...",
      "...BBBBBBBBBB...",
      "....BMMMMMMB....",
      "....BMMHHMMB....",
      "....BMMMMMMB....",
      "....BMMMMMMB....",
      "...BMMMSSSSMB...",
      "..BBBBBBBBBBBB..",
      ".BMMMMMMMMMMMMB.",
      ".BMMMMMMMMMMMMB.",
      ".BBBBBBBBBBBBBB.",
      "................"
    ],
    N: [
      "................",
      "......BBBB......",
      ".....BMMHMB.....",
      "....BMMMMMBB....",
      "...BMMHMMMMMB...",
      "..BMHHMMMMMMB...",
      "..BMHMMMMMMMB...",
      "..BBMMMMMMMMB...",
      "....BMMMMMMMB...",
      "....BMMMMMMMB...",
      "...BMMMSSSSMB...",
      "..BBBBBBBBBBBB..",
      ".BMMMMMMMMMMMMB.",
      ".BMMMMMMMMMMMMB.",
      ".BBBBBBBBBBBBBB.",
      "................"
    ],
    B: [
      "................",
      ".......BB.......",
      "......BMMB......",
      ".....BMMMMB.....",
      "....BMMHMMMB....",
      "....BMH.MMMB....",
      "....BMM.MMMB....",
      ".....BMMMMB.....",
      "....BBBBBBBB....",
      "....BMMMMMMB....",
      "...BMMMSSSSMB...",
      "..BBBBBBBBBBBB..",
      ".BMMMMMMMMMMMMB.",
      ".BMMMMMMMMMMMMB.",
      ".BBBBBBBBBBBBBB.",
      "................"
    ],
    Q: [
      "................",
      "..B....BB....B..",
      ".BMB..BMMB..BMB.",
      ".BMMBBMMMMBBMMB.",
      ".BMMMMMMMMMMMMB.",
      "..BMMHHHHHMMMB..",
      "...BMMMMMMMMB...",
      "....BMMMMMMB....",
      "...BMMMMMMMMB...",
      "..BMMMMMMMMMMB..",
      "..BMMMSSSSMMMB..",
      ".BBBBBBBBBBBBBB.",
      "BMMMMMMMMMMMMMMB",
      "BMMMMMMMMMMMMMMB",
      "BBBBBBBBBBBBBBBB",
      "................"
    ],
    K: [
      ".......BB.......",
      "......BMMB......",
      "....BBBBBBBB....",
      "......BMMB......",
      "....BBBBBBBB....",
      "...BMMMMMMMMB...",
      "..BMMHHHHHHMMB..",
      "..BMMMMMMMMMMB..",
      "...BMMMMMMMMB...",
      "....BMMMMMMB....",
      "...BMMMSSSSMB...",
      "..BBBBBBBBBBBB..",
      ".BMMMMMMMMMMMMB.",
      ".BMMMMMMMMMMMMB.",
      ".BBBBBBBBBBBBBB.",
      "................"
    ]
  };

  const PALETTES = {
    w: {
      B: "#101318",
      M: "#f0f4f8",
      H: "#ffffff",
      S: "#c3cad4"
    },
    b: {
      B: "#090a0d",
      M: "#2b303c",
      H: "#57606f",
      S: "#1a1c24"
    }
  };

  // Pre-generate SVG string cache
  const SVG_PIECE_CACHE = {};
  for (const [type, pattern] of Object.entries(PIXEL_PATTERNS)) {
    for (const color of ['w', 'b']) {
      const pal = PALETTES[color];
      let rects = "";

      for (let r = 0; r < 16; r++) {
        const row = pattern[r];
        let startC = -1;
        let currentColor = null;

        for (let c = 0; c < 16; c++) {
          const char = row[c];
          const fill = pal[char];

          if (fill) {
            if (currentColor === fill) {
              // continue span
            } else {
              if (currentColor !== null && startC !== -1) {
                rects += `<rect x="${startC}" y="${r}" width="${c - startC}" height="1" fill="${currentColor}"/>`;
              }
              startC = c;
              currentColor = fill;
            }
          } else {
            if (currentColor !== null && startC !== -1) {
              rects += `<rect x="${startC}" y="${r}" width="${c - startC}" height="1" fill="${currentColor}"/>`;
              startC = -1;
              currentColor = null;
            }
          }
        }
        if (currentColor !== null && startC !== -1) {
          rects += `<rect x="${startC}" y="${r}" width="${16 - startC}" height="1" fill="${currentColor}"/>`;
        }
      }

      const key = color === 'w' ? type.toUpperCase() : type.toLowerCase();
      SVG_PIECE_CACHE[key] = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="100%" height="100%" shape-rendering="crispEdges" class="chess-piece">${rects}</svg>`;
    }
  }

  // ==========================================================================
  // 2. 8-BIT WEB AUDIO API SYNTHESIZER
  // ==========================================================================
  class RetroAudio {
    constructor() {
      this.ctx = null;
      this.enabled = true;
    }

    init() {
      if (!this.ctx && (window.AudioContext || window.webkitAudioContext)) {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        this.ctx = new AudioCtx();
      }
      if (this.ctx && this.ctx.state === 'suspended') {
        this.ctx.resume();
      }
    }

    playMove() {
      if (!this.enabled) return;
      this.init();
      if (!this.ctx) return;

      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = 'square';

      const now = this.ctx.currentTime;
      osc.frequency.setValueAtTime(420, now);
      osc.frequency.exponentialRampToValueAtTime(180, now + 0.07);

      gain.gain.setValueAtTime(0.12, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.07);

      osc.connect(gain);
      gain.connect(this.ctx.destination);

      osc.start(now);
      osc.stop(now + 0.07);
    }

    playCapture() {
      if (!this.enabled) return;
      this.init();
      if (!this.ctx) return;

      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = 'sawtooth';

      const now = this.ctx.currentTime;
      osc.frequency.setValueAtTime(240, now);
      osc.frequency.exponentialRampToValueAtTime(60, now + 0.12);

      gain.gain.setValueAtTime(0.2, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.12);

      osc.connect(gain);
      gain.connect(this.ctx.destination);

      osc.start(now);
      osc.stop(now + 0.12);
    }

    playCheck() {
      if (!this.enabled) return;
      this.init();
      if (!this.ctx) return;

      const now = this.ctx.currentTime;
      const playTone = (freq, time, dur) => {
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.type = 'square';
        osc.frequency.setValueAtTime(freq, time);
        gain.gain.setValueAtTime(0.14, time);
        gain.gain.exponentialRampToValueAtTime(0.001, time + dur);
        osc.connect(gain);
        gain.connect(this.ctx.destination);
        osc.start(time);
        osc.stop(time + dur);
      };

      playTone(550, now, 0.09);
      playTone(820, now + 0.09, 0.14);
    }

    playVictory() {
      if (!this.enabled) return;
      this.init();
      if (!this.ctx) return;

      const now = this.ctx.currentTime;
      const notes = [261.6, 329.6, 392.0, 523.2]; // C4, E4, G4, C5
      notes.forEach((freq, idx) => {
        const t = now + idx * 0.1;
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(freq, t);
        gain.gain.setValueAtTime(0.18, t);
        gain.gain.exponentialRampToValueAtTime(0.001, t + 0.2);
        osc.connect(gain);
        gain.connect(this.ctx.destination);
        osc.start(t);
        osc.stop(t + 0.2);
      });
    }
  }

  // ==========================================================================
  // 3. COMPLETE CHESS STATE MACHINE
  // ==========================================================================
  class MiniChess {
    constructor() {
      this.reset();
    }

    reset() {
      this.board = [
        ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
        ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
        [null, null, null, null, null, null, null, null],
        [null, null, null, null, null, null, null, null],
        [null, null, null, null, null, null, null, null],
        [null, null, null, null, null, null, null, null],
        ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
        ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
      ];
      this.turn = 'w';
      this.castling = { K: true, Q: true, k: true, q: true };
      this.epSquare = null;
      this.halfmove = 0;
      this.fullmove = 1;
      this.history = [];
    }

    isWhite(p) { return p && p === p.toUpperCase(); }
    isBlack(p) { return p && p === p.toLowerCase(); }
    getPieceColor(p) { return !p ? null : (this.isWhite(p) ? 'w' : 'b'); }

    squareToCoords(sq) {
      const col = sq.charCodeAt(0) - 97;
      const row = 8 - parseInt(sq[1], 10);
      return { r: row, c: col };
    }

    coordsToSquare(r, c) {
      return String.fromCharCode(97 + c) + (8 - r);
    }

    getFen() {
      let fen = "";
      for (let r = 0; r < 8; r++) {
        let empty = 0;
        for (let c = 0; c < 8; c++) {
          const p = this.board[r][c];
          if (!p) {
            empty++;
          } else {
            if (empty > 0) { fen += empty; empty = 0; }
            fen += p;
          }
        }
        if (empty > 0) fen += empty;
        if (r < 7) fen += "/";
      }

      fen += ` ${this.turn} `;

      let cast = "";
      if (this.castling.K) cast += "K";
      if (this.castling.Q) cast += "Q";
      if (this.castling.k) cast += "k";
      if (this.castling.q) cast += "q";
      fen += cast ? cast : "-";

      fen += ` ${this.epSquare ? this.coordsToSquare(this.epSquare.r, this.epSquare.c) : "-"} `;
      fen += `${this.halfmove} ${this.fullmove}`;
      return fen;
    }

    isSquareAttacked(r, c, attackerColor) {
      // Pawn attacks
      const pRow = attackerColor === 'w' ? r + 1 : r - 1;
      const pawnChar = attackerColor === 'w' ? 'P' : 'p';
      if (pRow >= 0 && pRow < 8) {
        if (c - 1 >= 0 && this.board[pRow][c - 1] === pawnChar) return true;
        if (c + 1 < 8 && this.board[pRow][c + 1] === pawnChar) return true;
      }

      // Knight attacks
      const kMoves = [
        [-2, -1], [-2, 1], [-1, -2], [-1, 2],
        [1, -2], [1, 2], [2, -1], [2, 1]
      ];
      const knightChar = attackerColor === 'w' ? 'N' : 'n';
      for (const [dr, dc] of kMoves) {
        const nr = r + dr, nc = c + dc;
        if (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) {
          if (this.board[nr][nc] === knightChar) return true;
        }
      }

      // King attacks
      const kingChar = attackerColor === 'w' ? 'K' : 'k';
      for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
          if (dr === 0 && dc === 0) continue;
          const nr = r + dr, nc = c + dc;
          if (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) {
            if (this.board[nr][nc] === kingChar) return true;
          }
        }
      }

      // Straight line sliders (Rook / Queen)
      const straightDirs = [[-1, 0], [1, 0], [0, -1], [0, 1]];
      const rookChar = attackerColor === 'w' ? 'R' : 'r';
      const queenChar = attackerColor === 'w' ? 'Q' : 'q';
      for (const [dr, dc] of straightDirs) {
        let nr = r + dr, nc = c + dc;
        while (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) {
          const p = this.board[nr][nc];
          if (p) {
            if (p === rookChar || p === queenChar) return true;
            break;
          }
          nr += dr; nc += dc;
        }
      }

      // Diagonal sliders (Bishop / Queen)
      const diagDirs = [[-1, -1], [-1, 1], [1, -1], [1, 1]];
      const bishopChar = attackerColor === 'w' ? 'B' : 'b';
      for (const [dr, dc] of diagDirs) {
        let nr = r + dr, nc = c + dc;
        while (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) {
          const p = this.board[nr][nc];
          if (p) {
            if (p === bishopChar || p === queenChar) return true;
            break;
          }
          nr += dr; nc += dc;
        }
      }

      return false;
    }

    findKing(color) {
      const kChar = color === 'w' ? 'K' : 'k';
      for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
          if (this.board[r][c] === kChar) return { r, c };
        }
      }
      return null;
    }

    isInCheck(color = this.turn) {
      const kingPos = this.findKing(color);
      if (!kingPos) return false;
      const enemyColor = color === 'w' ? 'b' : 'w';
      return this.isSquareAttacked(kingPos.r, kingPos.c, enemyColor);
    }

    generatePseudoLegalMoves(color = this.turn) {
      const moves = [];
      const isW = color === 'w';
      const forward = isW ? -1 : 1;
      const startRow = isW ? 6 : 1;
      const promoRow = isW ? 0 : 7;
      const enemyColor = isW ? 'b' : 'w';

      for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
          const p = this.board[r][c];
          if (!p || this.getPieceColor(p) !== color) continue;

          const pieceType = p.toUpperCase();

          // PAWN
          if (pieceType === 'P') {
            const fRow = r + forward;
            if (fRow >= 0 && fRow < 8 && !this.board[fRow][c]) {
              if (fRow === promoRow) {
                for (const pr of ['Q', 'R', 'B', 'N']) {
                  moves.push({ from: { r, c }, to: { r: fRow, c }, promo: isW ? pr : pr.toLowerCase() });
                }
              } else {
                moves.push({ from: { r, c }, to: { r: fRow, c } });
                const f2Row = r + 2 * forward;
                if (r === startRow && !this.board[f2Row][c]) {
                  moves.push({ from: { r, c }, to: { r: f2Row, c }, isDoublePush: true });
                }
              }
            }

            for (const dc of [-1, 1]) {
              const capC = c + dc;
              if (capC >= 0 && capC < 8 && fRow >= 0 && fRow < 8) {
                const target = this.board[fRow][capC];
                if (target && this.getPieceColor(target) === enemyColor) {
                  if (fRow === promoRow) {
                    for (const pr of ['Q', 'R', 'B', 'N']) {
                      moves.push({ from: { r, c }, to: { r: fRow, c: capC }, promo: isW ? pr : pr.toLowerCase() });
                    }
                  } else {
                    moves.push({ from: { r, c }, to: { r: fRow, c: capC } });
                  }
                } else if (this.epSquare && this.epSquare.r === fRow && this.epSquare.c === capC) {
                  moves.push({ from: { r, c }, to: { r: fRow, c: capC }, isEnPassant: true });
                }
              }
            }
          }

          // KNIGHT
          else if (pieceType === 'N') {
            const kDeltas = [
              [-2, -1], [-2, 1], [-1, -2], [-1, 2],
              [1, -2], [1, 2], [2, -1], [2, 1]
            ];
            for (const [dr, dc] of kDeltas) {
              const nr = r + dr, nc = c + dc;
              if (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) {
                const target = this.board[nr][nc];
                if (!target || this.getPieceColor(target) === enemyColor) {
                  moves.push({ from: { r, c }, to: { r: nr, c: nc } });
                }
              }
            }
          }

          // BISHOP / ROOK / QUEEN
          else if (pieceType === 'B' || pieceType === 'R' || pieceType === 'Q') {
            const dirs = [];
            if (pieceType === 'R' || pieceType === 'Q') dirs.push([-1, 0], [1, 0], [0, -1], [0, 1]);
            if (pieceType === 'B' || pieceType === 'Q') dirs.push([-1, -1], [-1, 1], [1, -1], [1, 1]);

            for (const [dr, dc] of dirs) {
              let nr = r + dr, nc = c + dc;
              while (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) {
                const target = this.board[nr][nc];
                if (!target) {
                  moves.push({ from: { r, c }, to: { r: nr, c: nc } });
                } else {
                  if (this.getPieceColor(target) === enemyColor) {
                    moves.push({ from: { r, c }, to: { r: nr, c: nc } });
                  }
                  break;
                }
                nr += dr; nc += dc;
              }
            }
          }

          // KING
          else if (pieceType === 'K') {
            for (let dr = -1; dr <= 1; dr++) {
              for (let dc = -1; dc <= 1; dc++) {
                if (dr === 0 && dc === 0) continue;
                const nr = r + dr, nc = c + dc;
                if (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) {
                  const target = this.board[nr][nc];
                  if (!target || this.getPieceColor(target) === enemyColor) {
                    moves.push({ from: { r, c }, to: { r: nr, c: nc } });
                  }
                }
              }
            }

            // Castling
            const homeRow = isW ? 7 : 0;
            if (r === homeRow && c === 4 && !this.isInCheck(color)) {
              const canK = isW ? this.castling.K : this.castling.k;
              if (canK && !this.board[homeRow][5] && !this.board[homeRow][6]) {
                if (!this.isSquareAttacked(homeRow, 5, enemyColor) && !this.isSquareAttacked(homeRow, 6, enemyColor)) {
                  moves.push({ from: { r, c: 4 }, to: { r, c: 6 }, isCastling: 'K' });
                }
              }
              const canQ = isW ? this.castling.Q : this.castling.q;
              if (canQ && !this.board[homeRow][1] && !this.board[homeRow][2] && !this.board[homeRow][3]) {
                if (!this.isSquareAttacked(homeRow, 2, enemyColor) && !this.isSquareAttacked(homeRow, 3, enemyColor)) {
                  moves.push({ from: { r, c: 4 }, to: { r, c: 2 }, isCastling: 'Q' });
                }
              }
            }
          }
        }
      }
      return moves;
    }

    makeMove(m) {
      const fromP = this.board[m.from.r][m.from.c];
      const toP = this.board[m.to.r][m.to.c];
      const isW = this.turn === 'w';

      this.history.push({
        move: m,
        fromPiece: fromP,
        toPiece: toP,
        castling: { ...this.castling },
        epSquare: this.epSquare ? { ...this.epSquare } : null,
        halfmove: this.halfmove,
        fullmove: this.fullmove
      });

      this.board[m.from.r][m.from.c] = null;
      let placedPiece = m.promo || fromP;
      this.board[m.to.r][m.to.c] = placedPiece;

      if (m.isEnPassant) {
        const capRow = isW ? m.to.r + 1 : m.to.r - 1;
        this.board[capRow][m.to.c] = null;
      }

      if (m.isCastling === 'K') {
        this.board[m.to.r][7] = null;
        this.board[m.to.r][5] = isW ? 'R' : 'r';
      } else if (m.isCastling === 'Q') {
        this.board[m.to.r][0] = null;
        this.board[m.to.r][3] = isW ? 'R' : 'r';
      }

      // Invalidate castling rights on king or rook moves
      if (fromP === 'K') { this.castling.K = false; this.castling.Q = false; }
      if (fromP === 'k') { this.castling.k = false; this.castling.q = false; }
      if (m.from.r === 7 && m.from.c === 7) this.castling.K = false;
      if (m.from.r === 7 && m.from.c === 0) this.castling.Q = false;
      if (m.from.r === 0 && m.from.c === 7) this.castling.k = false;
      if (m.from.r === 0 && m.from.c === 0) this.castling.q = false;

      // Invalidate on rook captures
      if (m.to.r === 7 && m.to.c === 7) this.castling.K = false;
      if (m.to.r === 7 && m.to.c === 0) this.castling.Q = false;
      if (m.to.r === 0 && m.to.c === 7) this.castling.k = false;
      if (m.to.r === 0 && m.to.c === 0) this.castling.q = false;

      if (m.isDoublePush) {
        this.epSquare = { r: (m.from.r + m.to.r) / 2, c: m.from.c };
      } else {
        this.epSquare = null;
      }

      if (fromP.toUpperCase() === 'P' || toP) {
        this.halfmove = 0;
      } else {
        this.halfmove++;
      }

      if (this.turn === 'b') {
        this.fullmove++;
        this.turn = 'w';
      } else {
        this.turn = 'b';
      }
    }

    undoMove() {
      if (this.history.length === 0) return null;
      const last = this.history.pop();
      const m = last.move;
      const isW = this.turn === 'b';

      this.board[m.from.r][m.from.c] = last.fromPiece;
      this.board[m.to.r][m.to.c] = last.toPiece;

      if (m.isEnPassant) {
        const capRow = isW ? m.to.r + 1 : m.to.r - 1;
        this.board[capRow][m.to.c] = isW ? 'p' : 'P';
      }

      if (m.isCastling === 'K') {
        this.board[m.to.r][7] = isW ? 'R' : 'r';
        this.board[m.to.r][5] = null;
      } else if (m.isCastling === 'Q') {
        this.board[m.to.r][0] = isW ? 'R' : 'r';
        this.board[m.to.r][3] = null;
      }

      this.castling = last.castling;
      this.epSquare = last.epSquare;
      this.halfmove = last.halfmove;
      this.fullmove = last.fullmove;
      this.turn = isW ? 'w' : 'b';
      return m;
    }

    getLegalMoves() {
      const pseudo = this.generatePseudoLegalMoves(this.turn);
      const legal = [];
      for (const m of pseudo) {
        this.makeMove(m);
        const inCheck = this.isInCheck(this.turn === 'w' ? 'b' : 'w');
        this.undoMove();
        if (!inCheck) {
          legal.push(m);
        }
      }
      return legal;
    }

    moveToUci(m) {
      const fromStr = this.coordsToSquare(m.from.r, m.from.c);
      const toStr = this.coordsToSquare(m.to.r, m.to.c);
      const promo = m.promo ? m.promo.toLowerCase() : "";
      return fromStr + toStr + promo;
    }

    uciToMove(uci) {
      if (!uci || uci.length < 4) return null;
      const from = this.squareToCoords(uci.slice(0, 2));
      const to = this.squareToCoords(uci.slice(2, 4));
      const promo = uci.length > 4 ? uci[4] : null;

      const legals = this.getLegalMoves();
      for (const m of legals) {
        if (m.from.r === from.r && m.from.c === from.c && m.to.r === to.r && m.to.c === to.c) {
          if (!promo || (m.promo && m.promo.toLowerCase() === promo.toLowerCase())) {
            return m;
          }
        }
      }
      return null;
    }

    isGameOver() {
      const legals = this.getLegalMoves();
      if (legals.length === 0) {
        if (this.isInCheck(this.turn)) {
          return { gameOver: true, result: this.turn === 'w' ? '0-1' : '1-0', reason: 'CHECKMATE' };
        }
        return { gameOver: true, result: '1/2-1/2', reason: 'STALEMATE' };
      }
      if (this.halfmove >= 100) {
        return { gameOver: true, result: '1/2-1/2', reason: '50-MOVE DRAW' };
      }
      return { gameOver: false };
    }
  }

  // ==========================================================================
  // 4. OFFLINE BROWSER FALLBACK AI (MINIMAX + PIECE-SQUARE TABLES)
  // ==========================================================================
  const PIECE_VALUES = {
    P: 100, N: 320, B: 330, R: 500, Q: 900, K: 20000,
    p: -100, n: -320, b: -330, r: -500, q: -900, k: -20000
  };

  const PST_PAWN = [
    [0,  0,  0,  0,  0,  0,  0,  0],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [5,  5, 10, 25, 25, 10,  5,  5],
    [0,  0,  0, 20, 20,  0,  0,  0],
    [5, -5,-10,  0,  0,-10, -5,  5],
    [5, 10, 10,-20,-20, 10, 10,  5],
    [0,  0,  0,  0,  0,  0,  0,  0]
  ];

  const PST_KNIGHT = [
    [-50,-40,-30,-30,-30,-30,-40,-50],
    [-40,-20,  0,  0,  0,  0,-20,-40],
    [-30,  0, 10, 15, 15, 10,  0,-30],
    [-30,  5, 15, 20, 20, 15,  5,-30],
    [-30,  0, 15, 20, 20, 15,  0,-30],
    [-30,  5, 10, 15, 15, 10,  5,-30],
    [-40,-20,  0,  5,  5,  0,-20,-40],
    [-50,-40,-30,-30,-30,-30,-40,-50]
  ];

  function evaluateBoardLocal(chess) {
    let score = 0;
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const p = chess.board[r][c];
        if (!p) continue;
        score += PIECE_VALUES[p];
        if (p === 'P') score += PST_PAWN[r][c];
        else if (p === 'p') score -= PST_PAWN[7 - r][c];
        else if (p === 'N') score += PST_KNIGHT[r][c];
        else if (p === 'n') score -= PST_KNIGHT[7 - r][c];
      }
    }
    return score;
  }

  function offlineMinimax(chess, depth, alpha, beta, isMaximizing, nodes) {
    nodes.count++;
    if (depth === 0) {
      return { score: evaluateBoardLocal(chess) };
    }

    const legalMoves = chess.getLegalMoves();
    if (legalMoves.length === 0) {
      if (chess.isInCheck(chess.turn)) {
        return { score: isMaximizing ? -90000 + (4 - depth) : 90000 - (4 - depth) };
      }
      return { score: 0 };
    }

    // Sort captures first
    legalMoves.sort((a, b) => {
      const capA = chess.board[a.to.r][a.to.c] ? 1 : 0;
      const capB = chess.board[b.to.r][b.to.c] ? 1 : 0;
      return capB - capA;
    });

    let bestMove = legalMoves[0];

    if (isMaximizing) {
      let maxEval = -Infinity;
      for (const m of legalMoves) {
        chess.makeMove(m);
        const res = offlineMinimax(chess, depth - 1, alpha, beta, false, nodes);
        chess.undoMove();
        if (res.score > maxEval) {
          maxEval = res.score;
          bestMove = m;
        }
        alpha = Math.max(alpha, res.score);
        if (beta <= alpha) break;
      }
      return { score: maxEval, move: bestMove };
    } else {
      let minEval = Infinity;
      for (const m of legalMoves) {
        chess.makeMove(m);
        const res = offlineMinimax(chess, depth - 1, alpha, beta, true, nodes);
        chess.undoMove();
        if (res.score < minEval) {
          minEval = res.score;
          bestMove = m;
        }
        beta = Math.min(beta, res.score);
        if (beta <= alpha) break;
      }
      return { score: minEval, move: bestMove };
    }
  }

  // ==========================================================================
  // 5. UI CONTROLLER & EVENT COORDINATOR
  // ==========================================================================
  class PixelChessApp {
    constructor() {
      this.game = new MiniChess();
      this.audio = new RetroAudio();

      this.mode = 'human';       // 'human' | 'auto' | 'free'
      this.playerColor = 'w';    // 'w' | 'b'
      this.searchDepth = 3;
      this.isThinking = false;
      this.autoTimer = null;

      this.selectedCoords = null;
      this.legalTargets = [];
      this.lastMove = null;

      // DOM Elements
      this.elBoard = document.getElementById('chessboard');
      this.elStatus = document.getElementById('statusMessage');
      this.elEvalBar = document.getElementById('evalBarWhite');
      this.elEvalText = document.getElementById('evalScoreText');
      this.elMoveLog = document.getElementById('moveLogList');
      this.elStatNodes = document.getElementById('statNodes');
      this.elStatTime = document.getElementById('statTime');

      this.initBoardDOM();
      this.bindControls();
      this.render();
      this.setStatus("WHITE TO MOVE. SELECT A PIECE.");
    }

    initBoardDOM() {
      this.elBoard.innerHTML = "";
      this.squareElements = [];

      for (let r = 0; r < 8; r++) {
        this.squareElements[r] = [];
        for (let c = 0; c < 8; c++) {
          const sq = document.createElement('div');
          sq.className = `square ${(r + c) % 2 === 0 ? 'light' : 'dark'}`;
          sq.dataset.row = r;
          sq.dataset.col = c;
          sq.addEventListener('click', () => this.handleSquareClick(r, c));
          this.elBoard.appendChild(sq);
          this.squareElements[r][c] = sq;
        }
      }
    }

    bindControls() {
      // Mode buttons
      const btnModeHuman = document.getElementById('btnModeHuman');
      const btnModeAuto = document.getElementById('btnModeAuto');
      const btnModeFree = document.getElementById('btnModeFree');

      const setMode = (newMode) => {
        this.mode = newMode;
        btnModeHuman.classList.toggle('active', newMode === 'human');
        btnModeAuto.classList.toggle('active', newMode === 'auto');
        btnModeFree.classList.toggle('active', newMode === 'free');

        if (this.autoTimer) {
          clearTimeout(this.autoTimer);
          this.autoTimer = null;
        }

        if (newMode === 'auto') {
          this.triggerAiMove();
        } else if (newMode === 'human') {
          if (this.game.turn !== this.playerColor) {
            this.triggerAiMove();
          }
        }
      };

      btnModeHuman.addEventListener('click', () => setMode('human'));
      btnModeAuto.addEventListener('click', () => setMode('auto'));
      btnModeFree.addEventListener('click', () => setMode('free'));

      // Color selection
      const btnColorWhite = document.getElementById('btnColorWhite');
      const btnColorBlack = document.getElementById('btnColorBlack');

      btnColorWhite.addEventListener('click', () => {
        this.playerColor = 'w';
        btnColorWhite.classList.add('active');
        btnColorBlack.classList.remove('active');
        this.updateOrientation();
        this.render();
        if (this.mode === 'human' && this.game.turn !== 'w') {
          this.triggerAiMove();
        }
      });

      btnColorBlack.addEventListener('click', () => {
        this.playerColor = 'b';
        btnColorBlack.classList.add('active');
        btnColorWhite.classList.remove('active');
        this.updateOrientation();
        this.render();
        if (this.mode === 'human' && this.game.turn === 'w') {
          this.triggerAiMove();
        }
      });

      // Depth selector
      document.querySelectorAll('.depth-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
          document.querySelectorAll('.depth-btn').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          this.searchDepth = parseInt(btn.dataset.depth, 10);
        });
      });

      // Action buttons
      document.getElementById('btnAiMove').addEventListener('click', () => {
        this.triggerAiMove();
      });

      document.getElementById('btnReset').addEventListener('click', () => {
        if (this.autoTimer) clearTimeout(this.autoTimer);
        this.game.reset();
        this.selectedCoords = null;
        this.legalTargets = [];
        this.lastMove = null;
        this.elMoveLog.innerHTML = '<div class="log-placeholder">Game restarted. Ready.</div>';
        this.updateEvalGauge(0);
        this.setStatus("WHITE TO MOVE. SELECT A PIECE.");
        this.render();
      });

      document.getElementById('btnUndo').addEventListener('click', () => {
        if (this.isThinking) return;
        if (this.autoTimer) clearTimeout(this.autoTimer);

        if (this.mode === 'human') {
          // Revert 2 plies to give turn back to human
          this.game.undoMove();
          this.game.undoMove();
        } else {
          this.game.undoMove();
        }

        this.selectedCoords = null;
        this.legalTargets = [];
        this.lastMove = this.game.history.length > 0 ? this.game.history[this.game.history.length - 1].move : null;
        this.render();
        this.setStatus(`TURN: ${this.game.turn === 'w' ? 'WHITE' : 'BLACK'}`);
      });

      // Sound toggle
      const btnSound = document.getElementById('btnSoundToggle');
      btnSound.addEventListener('click', () => {
        this.audio.enabled = !this.audio.enabled;
        btnSound.textContent = this.audio.enabled ? 'ON' : 'OFF';
        btnSound.className = `sound-btn ${this.audio.enabled ? 'on' : 'off'}`;
      });
    }

    updateOrientation() {
      const isBlack = this.playerColor === 'b';
      for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
          const displayR = isBlack ? 7 - r : r;
          const displayC = isBlack ? 7 - c : c;
          this.squareElements[r][c].style.order = displayR * 8 + displayC;
        }
      }

      const files = isBlack ? ['H', 'G', 'F', 'E', 'D', 'C', 'B', 'A'] : ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'];
      const ranks = isBlack ? ['1', '2', '3', '4', '5', '6', '7', '8'] : ['8', '7', '6', '5', '4', '3', '2', '1'];

      document.querySelectorAll('.coords-files').forEach(el => {
        el.innerHTML = files.map(f => `<span>${f}</span>`).join('');
      });
      document.querySelectorAll('.coords-ranks').forEach(el => {
        el.innerHTML = ranks.map(r => `<span>${r}</span>`).join('');
      });
    }

    setStatus(text) {
      this.elStatus.textContent = text;
    }

    handleSquareClick(r, c) {
      if (this.isThinking) return;

      // In human mode, block clicks when it's AI's turn
      if (this.mode === 'human' && this.game.turn !== this.playerColor) {
        return;
      }

      // If already selected a square and clicked a legal target square
      if (this.selectedCoords) {
        const targetMove = this.legalTargets.find(m => m.to.r === r && m.to.c === c);
        if (targetMove) {
          this.executeMove(targetMove);
          this.selectedCoords = null;
          this.legalTargets = [];
          this.render();

          // Trigger AI if in human mode and game not over
          const over = this.game.isGameOver();
          if (!over.gameOver && this.mode === 'human') {
            setTimeout(() => this.triggerAiMove(), 150);
          }
          return;
        }
      }

      // Select piece of current turn's color
      const piece = this.game.board[r][c];
      if (piece && this.game.getPieceColor(piece) === this.game.turn) {
        this.selectedCoords = { r, c };
        const allLegal = this.game.getLegalMoves();
        this.legalTargets = allLegal.filter(m => m.from.r === r && m.from.c === c);
        this.audio.playMove();
        this.render();
      } else {
        this.selectedCoords = null;
        this.legalTargets = [];
        this.render();
      }
    }

    executeMove(move, isAi = false) {
      const isCapture = !!this.game.board[move.to.r][move.to.c] || move.isEnPassant;
      const uci = this.game.moveToUci(move);
      const plyNum = this.game.history.length + 1;

      this.game.makeMove(move);
      this.lastMove = move;

      // Audio feedback
      if (this.game.isInCheck(this.game.turn)) {
        this.audio.playCheck();
      } else if (isCapture) {
        this.audio.playCapture();
      } else {
        this.audio.playMove();
      }

      // Append to Move Log
      this.appendMoveToLog(uci, plyNum);

      // Check game over
      const over = this.game.isGameOver();
      if (over.gameOver) {
        this.setStatus(`GAME OVER: ${over.reason} (${over.result})`);
        this.audio.playVictory();
      } else {
        const turnStr = this.game.turn === 'w' ? 'WHITE' : 'BLACK';
        this.setStatus(`${turnStr} TO MOVE.`);
      }

      this.render();
    }

    appendMoveToLog(uci, plyNum) {
      if (this.game.history.length === 1) {
        this.elMoveLog.innerHTML = "";
      }

      const moveNum = Math.ceil(plyNum / 2);
      const isWhiteMove = plyNum % 2 !== 0;

      let row = document.getElementById(`moveRow-${moveNum}`);
      if (!row) {
        row = document.createElement('div');
        row.id = `moveRow-${moveNum}`;
        row.className = 'log-entry';
        row.innerHTML = `<span class="log-num">${moveNum}.</span><span class="log-moves"><span class="w-move">${uci}</span></span>`;
        this.elMoveLog.appendChild(row);
      } else {
        const movesSpan = row.querySelector('.log-moves');
        movesSpan.innerHTML += ` <span class="b-move">${uci}</span>`;
      }

      this.elMoveLog.scrollTop = this.elMoveLog.scrollHeight;
    }

    updateEvalGauge(scoreCp) {
      // Centipawn to winning percentage sigmoid curve
      const cp = Math.max(-1500, Math.min(1500, scoreCp));
      const winProb = 1 / (1 + Math.exp(-0.0035 * cp));
      const whitePct = Math.max(3, Math.min(97, Math.round(winProb * 100)));

      this.elEvalBar.style.height = `${whitePct}%`;
      const sign = scoreCp > 0 ? "+" : "";
      this.elEvalText.textContent = `${sign}${(scoreCp / 100.0).toFixed(1)}`;
    }

    async triggerAiMove() {
      if (this.isThinking) return;
      const over = this.game.isGameOver();
      if (over.gameOver) return;

      this.isThinking = true;
      this.setStatus("AI CALCULATING BEST MOVE...");

      const t0 = performance.now();
      const currentFen = this.game.getFen();

      try {
        // Attempt Python backend first (server.py on port 8000)
        const response = await fetch('/api/move', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ fen: currentFen, depth: this.searchDepth })
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();

        const elapsed = (performance.now() - t0) / 1000;
        this.elStatTime.textContent = `${elapsed.toFixed(2)}s`;
        this.elStatNodes.textContent = (data.nodes || 0).toLocaleString();

        if (data.score_cp !== undefined) {
          this.updateEvalGauge(data.score_cp);
        }

        const moveObj = this.game.uciToMove(data.move);
        if (moveObj) {
          this.executeMove(moveObj, true);
        }

      } catch (err) {
        // Python server not active -> Fallback to built-in browser engine
        const elapsedLocalStart = performance.now();
        const nodesObj = { count: 0 };
        const isMax = this.game.turn === 'w';

        const result = offlineMinimax(this.game, Math.min(3, this.searchDepth), -Infinity, Infinity, isMax, nodesObj);
        const elapsed = (performance.now() - elapsedLocalStart) / 1000;

        this.elStatTime.textContent = `${elapsed.toFixed(2)}s [LOCAL]`;
        this.elStatNodes.textContent = nodesObj.count.toLocaleString();
        this.updateEvalGauge(result.score || 0);

        if (result.move) {
          this.executeMove(result.move, true);
        }
      } finally {
        this.isThinking = false;

        // Auto mode chaining
        if (this.mode === 'auto' && !this.game.isGameOver().gameOver) {
          this.autoTimer = setTimeout(() => this.triggerAiMove(), 400);
        }
      }
    }

    render() {
      const inCheck = this.game.isInCheck(this.game.turn);
      const kingPos = inCheck ? this.game.findKing(this.game.turn) : null;

      for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
          const sq = this.squareElements[r][c];
          sq.className = `square ${(r + c) % 2 === 0 ? 'light' : 'dark'}`;
          sq.innerHTML = "";

          // Last move highlight
          if (this.lastMove) {
            if ((this.lastMove.from.r === r && this.lastMove.from.c === c) ||
                (this.lastMove.to.r === r && this.lastMove.to.c === c)) {
              sq.classList.add('last-move');
            }
          }

          // Selected square
          if (this.selectedCoords && this.selectedCoords.r === r && this.selectedCoords.c === c) {
            sq.classList.add('selected');
          }

          // In-check King highlight
          if (kingPos && kingPos.r === r && kingPos.c === c) {
            sq.classList.add('in-check');
          }

          // Piece rendering
          const piece = this.game.board[r][c];
          if (piece) {
            sq.innerHTML = SVG_PIECE_CACHE[piece] || "";
          }

          // Legal move indicator overlays
          const isTarget = this.legalTargets.find(m => m.to.r === r && m.to.c === c);
          if (isTarget) {
            if (piece || isTarget.isEnPassant) {
              const capBracket = document.createElement('div');
              capBracket.className = 'legal-capture';
              sq.appendChild(capBracket);
            } else {
              const dot = document.createElement('div');
              dot.className = 'legal-dot';
              sq.appendChild(dot);
            }
          }
        }
      }
    }
  }

  // Boot the app when DOM is ready
  window.addEventListener('DOMContentLoaded', () => {
    window.app = new PixelChessApp();
  });
})();
