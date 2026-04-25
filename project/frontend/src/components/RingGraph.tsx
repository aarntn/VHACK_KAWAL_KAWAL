import { useEffect, useRef, useState, useCallback } from 'react';
import * as d3 from 'd3-force';
import { Maximize2, RotateCcw } from 'lucide-react';
import { fetchRingGraph, type RingGraphNode, type RingGraphLink, type RingGraphResponse, type RingMeta } from '../api';

// ── palette ───────────────────────────────────────────────────────────────────
const TIER_COLOR: Record<string, string> = {
  high:   '#ef4444',
  medium: '#f59e0b',
  low:    '#facc15',
};
const TIER_LABEL: Record<string, string> = {
  high: 'High Risk', medium: 'Medium Risk', low: 'Low Risk',
};
const TIER_PILL_TINT: Record<string, string> = {
  high: 'rgba(239, 68, 68, 0.12)',
  medium: 'rgba(245, 158, 11, 0.12)',
  low: 'rgba(250, 204, 21, 0.12)',
};
// attribute-type colours (edge + node)
const ATTR_COLOR: Record<string, string> = {
  device:  '#3b82f6', // blue
  ip:      '#a855f7', // purple
  card:    '#2dd4bf', // teal
  unknown: '#6b7280',
};
const ATTR_LABEL: Record<string, string> = {
  device: 'Device ID', ip: 'IP Subnet', card: 'Card Prefix',
};
const ATTR_PILL_TINT: Record<string, string> = {
  device: 'rgba(59, 130, 246, 0.12)',
  ip: 'rgba(168, 85, 247, 0.12)',
  card: 'rgba(45, 212, 191, 0.12)',
  unknown: 'rgba(107, 114, 128, 0.12)',
};

// ── sim types ─────────────────────────────────────────────────────────────────
interface SimNode extends RingGraphNode, d3.SimulationNodeDatum {
  x: number; y: number;
}
interface SimLink extends d3.SimulationLinkDatum<SimNode> {
  weight: number;
  edge_type?: string;
}
interface Camera { scale: number; tx: number; ty: number }

// ── sizing helpers ────────────────────────────────────────────────────────────
function nodeRadius(n: SimNode): number {
  if (n.type === 'account')   return 7;
  if (n.type === 'attribute') return 9;
  return 7;
}

function tierOf(n: SimNode): string {
  return n.tier ?? 'low';
}

function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}

// ── draw ──────────────────────────────────────────────────────────────────────
function draw(
  ctx: CanvasRenderingContext2D,
  nodes: SimNode[],
  links: SimLink[],
  rings: RingMeta[],
  hoverId: string | null,
  connectedIds: Set<string>,
  activeFilter: string | null,   // tier filter
  attrFilter: string | null,     // attribute-type filter
  cam: Camera,
) {
  const { width: w, height: h } = ctx.canvas;
  ctx.clearRect(0, 0, w, h);

  ctx.save();
  ctx.translate(cam.tx, cam.ty);
  ctx.scale(cam.scale, cam.scale);

  const dimmed = hoverId !== null;

  // ── ring cluster halos ────────────────────────────────────────────────────
  // compute per-ring centroid from account nodes
  const ringCentroids = new Map<string, { x: number; y: number; count: number }>();
  for (const n of nodes) {
    if (n.type !== 'account' || !n.ring_id) continue;
    const c = ringCentroids.get(n.ring_id) ?? { x: 0, y: 0, count: 0 };
    c.x += n.x ?? 0; c.y += n.y ?? 0; c.count++;
    ringCentroids.set(n.ring_id, c);
  }
  for (const [rid, c] of ringCentroids) {
    if (c.count === 0) continue;
    const cx = c.x / c.count, cy = c.y / c.count;
    const ring = rings.find(r => r.ring_id === rid);
    if (!ring) continue;
    if (activeFilter && ring.tier !== activeFilter) continue;
    const color = TIER_COLOR[ring.tier] ?? '#6b7280';
    // subtle halo
    ctx.beginPath();
    ctx.arc(cx, cy, (ring.size ?? 5) * 9 + 20, 0, Math.PI * 2);
    ctx.fillStyle = color + '0a';
    ctx.fill();
    ctx.strokeStyle = color + '22';
    ctx.lineWidth = 1 / cam.scale;
    ctx.stroke();
    // ring label
    ctx.font = `bold ${11 / cam.scale}px Inter, sans-serif`;
    ctx.fillStyle = color + 'bb';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(
      `${rid.replace('ring_', '').slice(0, 8)}  •  ${(((ring.fraud_rate ?? 0) * 100)).toFixed(0)}% fraud`,
      cx, cy - (ring.size ?? 5) * 9 - 22,
    );
  }

  // ── edges ─────────────────────────────────────────────────────────────────
  for (const l of links) {
    const s = l.source as SimNode;
    const t = l.target as SimNode;
    if (!s.x || !t.x) continue;
    if (attrFilter && t.attr_type !== attrFilter) continue;
    if (activeFilter && tierOf(s) !== activeFilter && tierOf(t) !== activeFilter) continue;

    const connected = !dimmed || (connectedIds.has(s.id) && connectedIds.has(t.id));
    const edgeColor = ATTR_COLOR[l.edge_type ?? 'unknown'] ?? '#6b7280';
    ctx.beginPath();
    ctx.moveTo(s.x, s.y);
    ctx.lineTo(t.x, t.y);
    ctx.strokeStyle = connected
      ? edgeColor + '66'
      : edgeColor + '12';
    ctx.lineWidth = (connected ? 1.2 : 0.5) / cam.scale;
    ctx.stroke();
  }

  // ── nodes ─────────────────────────────────────────────────────────────────
  for (const n of nodes) {
    if (!n.x || !n.y) continue;
    if (activeFilter && tierOf(n) !== activeFilter) continue;
    if (attrFilter && n.type === 'attribute' && n.attr_type !== attrFilter) continue;

    const r       = nodeRadius(n);
    const isHov   = n.id === hoverId;
    const isConn  = connectedIds.has(n.id);
    const fade    = dimmed && !isHov && !isConn;
    const alpha   = fade ? '28' : 'dd';
    const rr      = r + (isHov ? 3 : 0);

    if (n.type === 'account') {
      const color = TIER_COLOR[tierOf(n)] ?? '#6b7280';
      ctx.beginPath();
      ctx.arc(n.x, n.y, rr, 0, Math.PI * 2);
      ctx.fillStyle = color + alpha;
      ctx.fill();
      if (isHov) {
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2 / cam.scale;
        ctx.stroke();
      }
    } else if (n.type === 'attribute') {
      // draw as rounded diamond (rotated square)
      const color = ATTR_COLOR[n.attr_type ?? 'unknown'];
      ctx.save();
      ctx.translate(n.x, n.y);
      ctx.rotate(Math.PI / 4);
      ctx.beginPath();
      ctx.rect(-rr * 0.72, -rr * 0.72, rr * 1.44, rr * 1.44);
      ctx.fillStyle = color + alpha;
      ctx.fill();
      ctx.strokeStyle = isHov ? '#fff' : color + (fade ? '30' : 'cc');
      ctx.lineWidth = (isHov ? 2 : 1) / cam.scale;
      ctx.stroke();
      ctx.restore();

      // attr-type icon letter inside
      if (!fade) {
        ctx.font = `bold ${8 / cam.scale}px monospace`;
        ctx.fillStyle = '#ffffffcc';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const icon = n.attr_type === 'device' ? 'D' : n.attr_type === 'ip' ? 'IP' : n.attr_type === 'card' ? 'C' : '?';
        ctx.fillText(icon, n.x, n.y);
      }
    }
  }

  ctx.restore();
}

// ── component ─────────────────────────────────────────────────────────────────
type RingGraphProps = {
  variant?: 'page' | 'embedded';
};

export default function RingGraph({ variant = 'page' }: RingGraphProps) {
  const canvasRef  = useRef<HTMLCanvasElement>(null);
  const simRef     = useRef<d3.Simulation<SimNode, SimLink> | null>(null);
  const nodesRef   = useRef<SimNode[]>([]);
  const linksRef   = useRef<SimLink[]>([]);
  const ringsRef   = useRef<RingMeta[]>([]);
  const rafRef     = useRef<number>(0);
  const camRef     = useRef<Camera>({ scale: 1, tx: 0, ty: 0 });
  const dragRef    = useRef<{ sx: number; sy: number; tx0: number; ty0: number } | null>(null);
  const adjacency  = useRef<Map<string, Set<string>>>(new Map());

  const hoverRef   = useRef<string | null>(null);
  const connRef    = useRef<Set<string>>(new Set());
  const filterRef  = useRef<string | null>(null);
  const attrFRef   = useRef<string | null>(null);

  const [summary, setSummary]        = useState<RingGraphResponse['summary'] | null>(null);
  const [loading, setLoading]        = useState(true);
  const [error, setError]            = useState<string | null>(null);
  const [hoverId, setHoverId]        = useState<string | null>(null);
  const [connectedIds, setConnected] = useState<Set<string>>(new Set());
  const [tooltip, setTooltip]        = useState<{ node: SimNode; sx: number; sy: number } | null>(null);
  const [activeFilter, setFilter]    = useState<string | null>(null);
  const [attrFilter, setAttrFilter]  = useState<string | null>(null);
  const [isDragging, setIsDragging]  = useState(false);

  useEffect(() => { hoverRef.current  = hoverId;      }, [hoverId]);
  useEffect(() => { connRef.current   = connectedIds; }, [connectedIds]);
  useEffect(() => { filterRef.current = activeFilter; }, [activeFilter]);
  useEffect(() => { attrFRef.current  = attrFilter;   }, [attrFilter]);

  const buildAdj = useCallback((nodes: SimNode[], links: SimLink[]) => {
    const adj = new Map<string, Set<string>>();
    for (const n of nodes) adj.set(n.id, new Set());
    for (const l of links) {
      const s = typeof l.source === 'object' ? (l.source as SimNode).id : l.source as string;
      const t = typeof l.target === 'object' ? (l.target as SimNode).id : l.target as string;
      adj.get(s)?.add(t);
      adj.get(t)?.add(s);
    }
    adjacency.current = adj;
  }, []);

  const fitView = useCallback((pad = 80) => {
    const canvas = canvasRef.current;
    const ns     = nodesRef.current;
    if (!canvas || ns.length === 0) return;
    const xs = ns.map(n => n.x), ys = ns.map(n => n.y);
    const bw = Math.max(...xs) - Math.min(...xs) || 1;
    const bh = Math.max(...ys) - Math.min(...ys) || 1;
    const s  = clamp(Math.min((canvas.width - pad * 2) / bw, (canvas.height - pad * 2) / bh), 0.2, 2.5);
    camRef.current = {
      scale: s,
      tx: (canvas.width  - bw * s) / 2 - Math.min(...xs) * s,
      ty: (canvas.height - bh * s) / 2 - Math.min(...ys) * s,
    };
  }, []);

  useEffect(() => {
    fetchRingGraph()
      .then(data => {
        setSummary(data.summary);
        ringsRef.current = data.rings ?? [];
        const canvas = canvasRef.current;
        if (!canvas) return;
        const W = canvas.width, H = canvas.height;
        const cx = W / 2, cy = H / 2;

        // group accounts by ring for pre-positioning
        const ringMembers = new Map<string, string[]>();
        for (const n of data.nodes) {
          if (n.type === 'account' && n.ring_id) {
            const arr = ringMembers.get(n.ring_id) ?? [];
            arr.push(n.id); ringMembers.set(n.ring_id, arr);
          }
        }
        const ringIds = [...ringMembers.keys()];
        const ringCenter = new Map<string, { x: number; y: number }>();
        ringIds.forEach((rid, i) => {
          const a = (i / ringIds.length) * Math.PI * 2 - Math.PI / 2;
          ringCenter.set(rid, {
            x: cx + Math.cos(a) * Math.min(W, H) * 0.32,
            y: cy + Math.sin(a) * Math.min(W, H) * 0.32,
          });
        });

        const simNodes: SimNode[] = data.nodes.map(n => {
          const rc = ringCenter.get(n.ring_id ?? '');
          const spread = n.type === 'attribute' ? 30 : 55;
          const p = rc
            ? { x: rc.x + (Math.random() - 0.5) * spread, y: rc.y + (Math.random() - 0.5) * spread }
            : { x: cx + (Math.random() - 0.5) * W * 0.5, y: cy + (Math.random() - 0.5) * H * 0.5 };
          return { ...n, x: p.x, y: p.y } as SimNode;
        });

        const idMap = new Map(simNodes.map(n => [n.id, n]));
        const simLinks: SimLink[] = (data.links as RingGraphLink[]).flatMap(l => {
          const s = idMap.get(l.source), t = idMap.get(l.target);
          return s && t ? [{ source: s, target: t, weight: l.weight, edge_type: l.edge_type }] : [];
        });

        nodesRef.current = simNodes;
        linksRef.current = simLinks;
        buildAdj(simNodes, simLinks);
        setTimeout(fitView, 300);

        const sim = d3.forceSimulation<SimNode, SimLink>(simNodes)
          .alphaDecay(0.022)
          .velocityDecay(0.38)
          .force('link', d3.forceLink<SimNode, SimLink>(simLinks)
            .id(n => n.id)
            .distance(n => (n as SimLink & { source: SimNode }).source?.type === 'attribute' ? 40 : 50)
            .strength(0.7),
          )
          .force('charge', d3.forceManyBody<SimNode>()
            .strength(n => n.type === 'attribute' ? -200 : -60)
            .distanceMax(250),
          )
          .force('collision', d3.forceCollide<SimNode>()
            .radius(n => nodeRadius(n) + 5)
            .iterations(3),
          )
          .force('center', d3.forceCenter(cx, cy).strength(0.03))
          .force('x', d3.forceX(cx).strength(0.006))
          .force('y', d3.forceY(cy).strength(0.006));

        simRef.current = sim;
      })
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false));

    return () => { simRef.current?.stop(); cancelAnimationFrame(rafRef.current); };
  }, [buildAdj, fitView]);

  // render loop — reads only refs
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx    = canvas?.getContext('2d');
    if (!ctx) return;
    const loop = () => {
      draw(ctx, nodesRef.current, linksRef.current, ringsRef.current,
        hoverRef.current, connRef.current, filterRef.current, attrFRef.current, camRef.current);
      rafRef.current = requestAnimationFrame(loop);
    };
    loop();
    return () => cancelAnimationFrame(rafRef.current);
  }, []);

  // wheel zoom — must be a non-passive native listener so preventDefault() works
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const mx   = (e.clientX - rect.left) * (canvas.width  / rect.width);
      const my   = (e.clientY - rect.top)  * (canvas.height / rect.height);
      const cam  = camRef.current;
      const zoom = Math.exp(-e.deltaY * 0.0015);
      const ns   = clamp(cam.scale * zoom, 0.15, 5.0);
      camRef.current = { scale: ns, tx: mx - ((mx - cam.tx) / cam.scale) * ns, ty: my - ((my - cam.ty) / cam.scale) * ns };
    };
    canvas.addEventListener('wheel', onWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', onWheel);
  }, []);

  // drag pan
  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    dragRef.current = { sx: e.clientX, sy: e.clientY, tx0: camRef.current.tx, ty0: camRef.current.ty };
    setIsDragging(true);
  }, []);
  const handleMouseUp = useCallback(() => { dragRef.current = null; setIsDragging(false); }, []);

  // hover + pan
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect  = canvas.getBoundingClientRect();
    const sx    = canvas.width  / rect.width;
    const sy    = canvas.height / rect.height;

    if (dragRef.current) {
      camRef.current = {
        scale: camRef.current.scale,
        tx: dragRef.current.tx0 + (e.clientX - dragRef.current.sx) * sx,
        ty: dragRef.current.ty0 + (e.clientY - dragRef.current.sy) * sy,
      };
      return;
    }
    const cam = camRef.current;
    const wx  = ((e.clientX - rect.left) * sx - cam.tx) / cam.scale;
    const wy  = ((e.clientY - rect.top)  * sy - cam.ty) / cam.scale;

    let closest: SimNode | null = null;
    let bestD = 24 / cam.scale;
    for (const n of nodesRef.current) {
      const d = Math.hypot((n.x ?? 0) - wx, (n.y ?? 0) - wy);
      if (d < bestD) { bestD = d; closest = n; }
    }
    setHoverId(closest?.id ?? null);
    setConnected(closest
      ? new Set([closest.id, ...(adjacency.current.get(closest.id) ?? [])])
      : new Set());
    setTooltip(closest
      ? { node: closest, sx: e.clientX - rect.left, sy: e.clientY - rect.top }
      : null);
  }, []);

  const handleMouseLeave = useCallback(() => {
    dragRef.current = null; setIsDragging(false);
    setHoverId(null); setConnected(new Set()); setTooltip(null);
  }, []);

  // ── render ────────────────────────────────────────────────────────────────
  const isEmbedded = variant === 'embedded';

  return (
    <div style={{
      minHeight: isEmbedded ? 'auto' : '100vh',
      background: isEmbedded ? 'transparent' : '#0c0e12',
      padding: isEmbedded ? 0 : '32px 24px',
      fontFamily: "'Inter Tight', Inter, sans-serif",
    }}>
      {!isEmbedded && (
        <>
          <h1 style={{ color: '#f7f7f7', fontSize: 22, fontWeight: 600, margin: '0 0 4px' }}>
            Fraud Ring Intelligence
          </h1>
          <p style={{ color: '#94979c', fontSize: 14, lineHeight: 1.45, margin: '0 0 20px' }}>
            Bipartite account–attribute graph. Accounts (circles) are connected through shared devices, IP subnets, and card prefixes (diamonds). Clusters = detected fraud rings.
            <span style={{ color: '#94979c', marginLeft: 8 }}>Scroll to zoom · Drag to pan · Hover to inspect</span>
          </p>
        </>
      )}

      {/* stats */}
      {summary && !isEmbedded && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 14, marginBottom: 24 }}>
          {[
            { label: 'Total Rings',       value: summary.total_rings,             color: '#f7f7f7' },
            { label: 'High Risk',         value: summary.high_risk_rings,         color: '#f7f7f7' },
            { label: 'Medium Risk',       value: summary.medium_risk_rings,       color: '#f7f7f7' },
            { label: 'Accounts in Rings', value: summary.total_accounts_in_rings, color: '#f7f7f7' },
            {
              label: 'Graph Evidence',
              value: summary.evidence_links_available ? 'Exact' : 'Summary',
              color: '#f7f7f7',
            },
          ].map(s => (
            <div key={s.label} style={{ background: 'transparent', border: '1px solid #22262f', borderRadius: 12, padding: '18px 20px' }}>
              <div style={{ fontSize: 14, lineHeight: '20px', fontWeight: 500, color: '#cecfd2', marginBottom: 12 }}>{s.label}</div>
              <div style={{ fontSize: 32, lineHeight: '38px', fontWeight: 600, color: s.color }}>{s.value}</div>
            </div>
          ))}
        </div>
      )}

      <div style={{ border: '1px solid #22262f', borderRadius: 12, background: '#13161b', padding: 14 }}>
        {/* toolbar */}
        <div style={{ display: 'flex', gap: 8, marginBottom: 14, flexWrap: 'wrap', alignItems: 'center' }}>
          {/* risk tier filter */}
          <span style={{ fontSize: 11, color: '#94979c', marginRight: 2 }}>Risk:</span>
          {(['all', 'high', 'medium', 'low'] as const).map(tier => (
            <button key={tier} onClick={() => setFilter(tier === 'all' ? null : tier)}
              style={{
                minHeight: 36,
                padding: '8px 12px',
                borderRadius: 8,
                border: '1px solid',
                fontSize: 12,
                fontWeight: 500,
                cursor: 'pointer',
                transition: 'background 160ms ease, border-color 160ms ease, color 160ms ease, box-shadow 160ms ease',
                borderColor: tier === 'all'
                  ? (activeFilter === null ? '#373a41' : '#292c32e6')
                  : ((activeFilter === tier ? TIER_COLOR[tier] + '66' : '#292c32e6')),
                background: (tier === 'all' ? activeFilter === null : activeFilter === tier)
                  ? (tier === 'all' ? '#22262f' : (TIER_PILL_TINT[tier] ?? '#15181f'))
                  : '#13161b',
                color: tier === 'all'
                  ? (activeFilter === null ? '#cecfd2' : '#94979c')
                  : (activeFilter === tier ? (TIER_COLOR[tier] ?? '#cecfd2') : '#cecfd2'),
                boxShadow: (tier === 'all' ? activeFilter === null : activeFilter === tier)
                  ? 'inset 0 0 0 1px rgba(247, 247, 247, 0.03)'
                  : 'none',
              }}>
              {tier === 'all' ? 'All' : TIER_LABEL[tier]}
            </button>
          ))}

          <span style={{ fontSize: 11, color: '#94979c', marginLeft: 8, marginRight: 2 }}>Attribute:</span>
          {(['all', 'device', 'ip', 'card'] as const).map(at => (
            <button key={at} onClick={() => setAttrFilter(at === 'all' ? null : at)}
              style={{
                minHeight: 36,
                padding: '8px 12px',
                borderRadius: 8,
                border: '1px solid',
                fontSize: 12,
                fontWeight: 500,
                cursor: 'pointer',
                transition: 'background 160ms ease, border-color 160ms ease, color 160ms ease, box-shadow 160ms ease',
                borderColor: at === 'all'
                  ? (attrFilter === null ? '#373a41' : '#292c32e6')
                  : (attrFilter === at ? (ATTR_COLOR[at] ?? '#292c32e6') + '66' : '#292c32e6'),
                background: (at === 'all' ? attrFilter === null : attrFilter === at)
                  ? (at === 'all' ? '#22262f' : (ATTR_PILL_TINT[at] ?? '#15181f'))
                  : '#13161b',
                color: at === 'all'
                  ? (attrFilter === null ? '#cecfd2' : '#94979c')
                  : (attrFilter === at ? (ATTR_COLOR[at] ?? '#cecfd2') : '#cecfd2'),
                boxShadow: (at === 'all' ? attrFilter === null : attrFilter === at)
                  ? 'inset 0 0 0 1px rgba(247, 247, 247, 0.03)'
                  : 'none',
              }}>
              {at === 'all' ? 'All' : ATTR_LABEL[at]}
            </button>
          ))}

          <div style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
            <button onClick={() => fitView()}
              style={{ minHeight: 36, padding: '8px 12px', borderRadius: 8, border: '1px solid #292c32e6', background: '#13161b', color: '#cecfd2', fontSize: 12, fontWeight: 500, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 8 }}>
              <Maximize2 size={14} />
              Fit
            </button>
            <button onClick={() => { camRef.current = { scale: 1, tx: 0, ty: 0 }; }}
              style={{ minHeight: 36, padding: '8px 12px', borderRadius: 8, border: '1px solid #292c32e6', background: '#13161b', color: '#cecfd2', fontSize: 12, fontWeight: 500, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 8 }}>
              <RotateCcw size={14} />
              Reset
            </button>
          </div>
        </div>

        {/* canvas */}
        <div style={{ position: 'relative', background: '#0c0e12', border: '1px solid #22262f', borderRadius: 12, overflow: 'hidden' }}>
        {loading && (
          <div style={{
            position: 'absolute',
            bottom: 36,
            left: '50%',
            transform: 'translateX(-50%)',
            background: '#13161be6',
            border: '1px solid #292c32e6',
            borderRadius: 20,
            padding: '4px 14px',
            color: '#94979c',
            fontSize: 11,
            pointerEvents: 'none',
            zIndex: 2,
          }}>
            Computing layout…
          </div>
        )}
        {error && (
          <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#ef4444', fontSize: 13 }}>
            {error}
          </div>
        )}
        <canvas
          ref={canvasRef}
          width={1100} height={640}
          style={{ display: 'block', width: '100%', cursor: isDragging ? 'grabbing' : hoverId ? 'pointer' : 'grab' }}
          onMouseMove={handleMouseMove}
          onMouseDown={handleMouseDown}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeave}
        />

        {/* tooltip */}
        {tooltip && (() => {
          const n = tooltip.node;
          const tier = tierOf(n);
          return (
            <div style={{
              position: 'absolute',
              left: Math.min(tooltip.sx + 16, (canvasRef.current?.clientWidth ?? 900) - 220),
              top:  Math.max(tooltip.sy - 10, 8),
              pointerEvents: 'none',
              background: '#13161bf2',
              border: '1px solid #292c32e6',
              borderRadius: 12,
              padding: '12px 14px',
              fontSize: 12,
              color: '#cecfd2',
              minWidth: 220,
              boxShadow: '0 18px 40px rgba(0,0,0,0.28)',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
                <span style={{
                  width: 10,
                  height: 10,
                  borderRadius: n.type === 'account' ? '50%' : 2,
                  transform: n.type === 'account' ? 'none' : 'rotate(45deg)',
                  background: n.type === 'attribute' ? ATTR_COLOR[n.attr_type ?? 'unknown'] : TIER_COLOR[tier] ?? '#94979c',
                  display: 'inline-block',
                  flexShrink: 0,
                }} />
                <div>
                  <div style={{ fontWeight: 500, marginBottom: 2, fontSize: 13, color: '#f7f7f7' }}>
                    {n.type === 'account' ? 'Account' : 'Attribute'}
                  </div>
                  <div style={{ color: '#94979c', fontSize: 11, lineHeight: '14px' }}>
                    {n.id.replace('attr_', '').slice(0, 22)}
                  </div>
                </div>
              </div>
              {n.type === 'account' ? (
                <>
                  <div style={{ marginBottom: 5, color: '#cecfd2' }}>Ring score <strong style={{ color: '#f7f7f7', fontWeight: 500 }}>{(n.ring_score ?? 0).toFixed(3)}</strong></div>
                  <div style={{ marginBottom: 5, color: '#cecfd2' }}>Risk <strong style={{ color: TIER_COLOR[tier], fontWeight: 500 }}>{TIER_LABEL[tier]}</strong></div>
                  <div style={{ color: '#94979c', fontSize: 11 }}>Ring {n.ring_id?.replace('ring_', '').slice(0, 8)}</div>
                </>
              ) : (
                <>
                  <div style={{ marginBottom: 5, color: '#cecfd2' }}>Type <strong style={{ color: ATTR_COLOR[n.attr_type ?? 'unknown'], fontWeight: 500 }}>{ATTR_LABEL[n.attr_type ?? 'unknown'] ?? n.attr_type}</strong></div>
                  <div style={{ marginBottom: 5, color: '#cecfd2' }}>Value <strong style={{ color: '#f7f7f7', fontWeight: 500 }}>{n.display}</strong></div>
                  <div style={{ marginBottom: 5, color: '#cecfd2' }}>Ring score <strong style={{ color: '#f7f7f7', fontWeight: 500 }}>{(n.ring_score ?? 0).toFixed(3)}</strong></div>
                  <div style={{ color: '#94979c', fontSize: 11 }}>Ring {n.ring_id?.replace('ring_', '').slice(0, 8)}</div>
                </>
              )}
            </div>
          );
        })()}

          <div style={{ position: 'absolute', bottom: 10, right: 14, fontSize: 10, color: '#374151', userSelect: 'none' }}>
            scroll to zoom · drag to pan
          </div>
        </div>

        {/* legend */}
        <div style={{ marginTop: 14, display: 'flex', gap: 20, flexWrap: 'wrap', alignItems: 'center' }}>
          <span style={{ fontSize: 11, color: '#6b7280' }}>Accounts:</span>
          {Object.entries(TIER_COLOR).map(([tier, color]) => (
            <span key={tier} style={{ fontSize: 11, color: '#8c909f', display: 'flex', alignItems: 'center', gap: 5 }}>
              <span style={{ width: 10, height: 10, borderRadius: '50%', background: color, display: 'inline-block' }} />
              {TIER_LABEL[tier]}
            </span>
          ))}
          <span style={{ fontSize: 11, color: '#6b7280', marginLeft: 8 }}>Attributes (◆):</span>
          {Object.entries(ATTR_COLOR).filter(([k]) => k !== 'unknown').map(([at, color]) => (
            <span key={at} style={{ fontSize: 11, color: '#8c909f', display: 'flex', alignItems: 'center', gap: 5 }}>
              <span style={{ width: 10, height: 10, background: color, display: 'inline-block', transform: 'rotate(45deg)' }} />
              {ATTR_LABEL[at]}
            </span>
          ))}
          <span style={{ fontSize: 11, color: '#6b7280', marginLeft: 8 }}>
            Links show exact observed evidence when available.
          </span>
        </div>

      </div>
    </div>
  );
}
