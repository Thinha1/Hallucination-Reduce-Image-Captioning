// Stable color per YOLO label, purely a hash of the string -- same label always gets the same
// bounding-box color across a session.
export function labelColor(label: string): string {
  let hash = 0;
  for (let i = 0; i < label.length; i++) {
    hash = (hash * 31 + label.charCodeAt(i)) >>> 0;
  }
  const hue = hash % 360;
  return `hsl(${hue}, 72%, 45%)`;
}

// Thresholds calibrated to this project's ClipScorer, which returns a raw CLIP cosine
// similarity (typically ~0.20-0.40 for a caption that actually matches the image) rather than
// a 0-1 confidence -- so "good" sits well below 0.8.
export function clipScoreColor(score: number): string {
  if (score >= 0.34) return "#16a34a"; // green: strong match
  if (score >= 0.27) return "#ca8a04"; // amber: plausible
  return "#dc2626"; // red: weak match
}
