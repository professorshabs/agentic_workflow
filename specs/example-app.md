# Spec: One-Page House Ad (Lightweight)

**Purpose:** A single page that sells one property and gets the viewer to contact the agent or owner. It works as a web page or a printable A4 flyer.

## Content (in order, top to bottom)

1. **Headline**: Short and specific, e.g. "Bright 3-Bed Family Home, Quiet Street".
2. **Hero photo**: One strong exterior shot. Full width, under 200 KB.
3. **Price and status**: Asking price or monthly rent, and "For Sale" or "To Let".
4. **Key facts strip**: Bedrooms, bathrooms, parking, floor size (m²), erf/stand size.
5. **Description**: 60-80 words covering the best features, the neighbourhood feel, and one standout selling point.
6. **Feature list**: 5-8 short bullets, e.g. garden, solar or backup power, security, fibre-ready, pet-friendly.
7. **Photo strip**: 3-4 small thumbnails of the lounge, kitchen, bedroom and garden. Optional.
8. **Contact block** (the most prominent element after the headline):
   - Name (and agency, if any)
   - Phone, tappable on mobile (`tel:` link)
   - WhatsApp link (optional but high-converting)
   - Email
   - Viewing details, e.g. "By appointment" or "Open house Sat 10:00-12:00"
9. **Footer**: Suburb/area (not the full address, for privacy), reference number, and a short disclaimer such as "Details subject to change."

## Design

- Single column, mobile-first, max width about 720 px, and A4-friendly for print.
- One accent color, one font family, and plenty of white space.
- Body text at 16 px or larger, with high contrast.
- Contact button sticky or repeated at the bottom on mobile.

## Technical (if web)

- One self-contained HTML file, with inline CSS and no frameworks or JavaScript.
- Images compressed (WebP or JPG), total page weight under 1 MB, and alt text on every image.
- Basic SEO and sharing tags: title, meta description, and Open Graph image so it previews well on WhatsApp.
- Print stylesheet so it saves cleanly to one A4 page.

## Acceptance Criteria

- The viewer can see what it is, the price, and how to contact someone within 5 seconds, without scrolling on mobile.
- Phone and WhatsApp links work with one tap.
- Loads in under 2 seconds on a mid-range mobile connection.
- Prints to exactly one page.


