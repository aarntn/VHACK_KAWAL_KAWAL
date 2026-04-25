# Kawal Dashboard Design System

This document is inferred from the Figma file `KAWAL` and the referenced desktop node:
`https://www.figma.com/design/V4R4HLUSQPMrFJl7v1DYFz/KAWAL?node-id=31008-37585&m=dev`

It is not a formal token export. It is a practical source of truth for keeping the implemented dashboard visually consistent with the Figma direction.

## Product Feel

Kawal should feel like a focused fraud-operations control plane:
- Dark, restrained, and high-trust
- Dense enough for operators, but not cluttered
- Quiet by default, with status colors reserved for decisions and health
- Executive summary first, evidence second

The UI should read as a modern B2B operations console rather than a marketing dashboard.

## Layout Principles

- Use a fixed left navigation rail around `296px` wide.
- Main content should feel centered within a max working width near `1080px`.
- Keep generous outer page padding around `32px`.
- Use a clear vertical rhythm with `24px` spacing between major sections.
- Overview should follow this order:
  1. Primary KPI row
  2. Secondary KPI row
  3. Decision mix summary
  4. Transactions table

## Color System

Primary neutrals inferred from Figma:
- App background: `#0C0E12`
- Elevated surface: `#13161B`
- Active nav / stronger surface: `#22262F`
- Border: `#22262F` to `#373A41`
- Primary text: `#F7F7F7`
- Secondary text: `#CECFD2`
- Muted text: `#94979C`

Semantic colors:
- Success green: `#17B26A`
- Warning amber: `#F79009`
- Danger red: `#F04438`

Badge surfaces:
- Approve badge bg: `#031F14`
- Approve badge border: `#053321`
- Flag badge bg: `#2C1C04`
- Flag badge border: `rgba(128, 82, 12, 0.25)`
- Block badge bg: `#2E0B05`
- Block badge border: `#4B130A`

Guideline:
- Use semantic color sparingly.
- Most of the screen should remain neutral.
- Avoid blue-heavy gradients or glowing effects on core surfaces.

## Typography

Figma uses an `Inter Tight` style. In implementation, use the closest available product font consistently.

Type hierarchy:
- Page title: around `30px`, semibold
- Card title: `16px`, semibold
- KPI value: `30px+`, semibold
- Body/supporting text: `14px`
- Table header: `12px`, semibold
- Table cell: `14px`

Typography rules:
- Tight hierarchy, not oversized
- Short supporting copy
- Favor semibold labels over excessive capitalization
- Keep tables and metadata compact

## Radius, Border, and Depth

- Standard card radius: `12px`
- Button radius: `8px`
- Pills/badges: fully rounded
- Default border weight: `1px`
- Shadow should be minimal and quiet

Avoid:
- Large soft glows
- Glassmorphism-heavy blur treatments
- Highly saturated outlines on default cards

## Navigation

Sidebar behavior:
- Brand at top with simple mark and wordmark
- Nav items are compact, left-aligned, and icon-led
- Active item uses a stronger filled neutral surface
- Inactive items should stay visually quiet

Navigation labels:
- Overview
- Transactions
- Fraud Ring
- System Health
- API Integration

## Cards

Card anatomy:
- Title at top left
- Optional overflow/kebab icon at top right
- Large KPI/value below
- One concise supporting sentence below the value

Use wider cards for the most important operational metrics:
- Processing Latency (p95)
- Risk Engine Health

Use smaller cards for business KPIs:
- Transactions Reviewed
- Estimated Fraud Prevented
- False Positive Rate

## Decision Mix

Decision mix should be presented as:
- One horizontal stacked distribution bar
- Three legend cards beneath it
- Each legend card contains:
  - colored dot
  - label
  - transaction count
  - percentage

This section should feel like a compact analytical summary rather than a chart-heavy panel.

## Tables

Transaction table conventions:
- Dark table surface with subtle header row contrast
- Thin dividers only
- No zebra striping
- Hover state should be subtle
- Columns should remain concise and scannable

Preferred transaction overview columns:
- User
- Transaction ID
- Timestamp
- Amount
- Risk Score
- Decision

Decision values should use compact status badges, not plain text.

## Buttons and Controls

Buttons should be low-noise secondary controls by default:
- Neutral dark fill
- Thin border
- Compact height
- Minimal shadow

Primary visible action on overview:
- Refresh

Do not overuse action buttons in KPI cards unless they actually open real actions.

## Content Strategy

Overview content should mix two scopes clearly:
- Aggregated operating metrics
- Recent transaction evidence

When scopes differ, label them honestly in supporting text instead of implying that all values come from the same dataset.

## Implementation Notes

When implementing future screens:
- Keep the dark neutral palette stable
- Reuse the same card spacing and radii
- Preserve the narrow operational tone
- Prefer clarity over decoration
- Use semantic colors only for meaning, never as default ornament

If a new UI element feels louder than the data it presents, it is probably off-system.
