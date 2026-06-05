# Lightweight Template Picks — v2

> **What changed vs v1**
> - `3_job-board`: Remix → B (was T3); T3 → C. Reason: T3 is "Next +
>   opinions", same philosophy as A. Remix gives a genuinely different
>   loader/action model.
> - `4_forum`: Django + HTMX promoted from C → B; T3 demoted to C.
>   Reason: Django is the more "natural" forum solution; T3 was redundant
>   with the SvelteKit-vs-batteries-included contrast we want.
> - `6_chat`: SvelteKit + Supabase Realtime promoted from C → A; Phoenix
>   LiveView demoted A → B. Reason: A position should reflect the most
>   *practical* lightweight starter, not the most technically pure one.
> - `7_cloud-storage`: Next.js + shadcn/ui promoted B → A; Refine demoted
>   A → B. Reason: this task is product-front-end + admin, not pure
>   admin; Refine over-optimizes the admin half.
> - `8_ecommerce`: plain Next.js + Stripe Checkout is the new A;
>   Next.js Commerce is now B; Astro + Snipcart unchanged at C; Medusa
>   moved out of the top 3 (mentioned as a heavier alternative). Reason:
>   for a 7-page benchmark, plain-Next is the cleanest starter; Commerce
>   is a "reference implementation" with provider coupling.
> - `10_streaming_music-streaming`: A/B swapped — Vite + React SPA is
>   now A, Next.js is B. Reason: a player-centric app benefits more from
>   a persistently-mounted SPA shell than from per-route SSR.
>
> Unchanged from v1: `1_newsletter`, `2_real-estate`, `5_travel-booking`,
> `9_project-management`.

A short legend used throughout:
- **SSG** = pre-rendered static site, **SSR** = rendered per request,
  **SPA** = client-rendered, **Islands** = mostly static + small JS.
- "Lightweight" here means: scaffolds from a single CLI command, has a
  small runtime footprint, and is a sensible *starting point* (not the
  most technically optimal architecture).

---

## 1_newsletter — blog / newsletter (mostly content, light interactivity)

### Pick A — Astro (content-first, islands)
```bash
npm create astro@latest -- --template blog
```
- **Why**: Built for exactly this shape (posts, tags, categories, RSS).
  MDX out of the box, 0 JS shipped by default.
- **Pros**: Tiny payloads, great Lighthouse, built-in content collections,
  trivial to deploy to any static host.
- **Cons**: Interactive bits need explicit `client:*` directives; comments
  / search need a third-party (Algolia, Pagefind) or an island.

### Pick B — Eleventy (11ty)
```bash
npm init -y && npm install --save-dev @11ty/eleventy
npx @11ty/eleventy --serve
```
- **Why**: Pure SSG, no client framework at all. Smallest possible bundle.
- **Pros**: Zero-runtime; trivial Markdown/Nunjucks; very fast builds;
  dirt-cheap hosting.
- **Cons**: No component model; interactive features (search, contact
  form) require hand-rolled JS or a backend webhook.

### Pick C — Next.js (App Router) with MDX
```bash
npx create-next-app@latest --ts --tailwind --eslint --app
```
- **Why**: Use SSG for posts and SSR/route handlers for the contact form
  and search; one stack for all 9 pages.
- **Pros**: Future-proof if the newsletter grows into a SaaS; Vercel
  preview deploys; built-in image optimization.
- **Cons**: Heavier baseline than Astro/11ty; you ship React even on
  static-feeling pages.

---

## 2_real-estate — listings + auth + map view

### Pick A — Next.js (App Router) + Tailwind
```bash
npx create-next-app@latest --ts --tailwind --eslint --app
```
- **Why**: SEO-critical listing pages benefit from SSR + ISR; auth and
  search are first-class via Route Handlers.
- **Pros**: Best ecosystem for real-estate-style hybrid (Mapbox/Leaflet
  components, image optimization, dynamic `[slug]` routes for properties).
- **Cons**: You'll bring your own ORM/auth (NextAuth, Prisma); cold
  starts on serverless if self-hosted.

### Pick B — Nuxt 3 (Vue)
```bash
npx nuxi@latest init real-estate
```
- **Why**: Same SSR + hybrid story as Next, but Vue's templating is
  friendly for marketing-heavy listing cards.
- **Pros**: File-based routing, auto-imports, `useFetch` makes data
  loading trivial; Nitro deploys almost anywhere.
- **Cons**: Smaller ecosystem of admin/UI kits than React; map libs tend
  to ship React-first wrappers.

### Pick C — Astro + a few React/Vue islands
```bash
npm create astro@latest
```
- **Why**: Listings and detail pages are mostly content; only the map
  and filters need to be interactive — the islands model fits.
- **Pros**: Minimal JS, great SEO, multi-framework islands.
- **Cons**: Auth + protected dashboard areas are clunkier than Next/Nuxt;
  needs a separate backend (Supabase, Pocketbase) for sign-in.

---

## 3_job-board — marketing landing + applicant dashboard

### Pick A — Next.js (App Router)
```bash
npx create-next-app@latest --ts --tailwind --eslint --app
```
- **Why**: Public job pages want SSR + good SEO; the dashboard wants
  client-side state. Next does both in one repo.
- **Pros**: Server Components for job listings, parallel routes for the
  dashboard sub-sections (Messages, Applications, Settings).
- **Cons**: Server Components have a steep mental model; you bring your
  own auth/ORM.

### Pick B — Remix
```bash
npx create-react-router@latest
```
- **Why**: A genuinely different philosophy from Next — loaders/actions
  match form-heavy flows (apply, message, edit profile) without writing
  API routes, and progressive enhancement means forms work without JS.
- **Pros**: Nested routes are perfect for dashboard sub-pages; great
  error/pending UX; web-platform-first mental model.
- **Cons**: Smaller plugin/UI-kit ecosystem than Next; non-Node deploys
  need adapters.

### Pick C — T3 stack (Next + tRPC + Prisma + NextAuth + Tailwind)
```bash
npm create t3-app@latest
```
- **Why**: Same rendering philosophy as A, but with a typed end-to-end
  CRUD layer pre-wired — useful if you want to model applicants /
  employers / postings as a typed schema from day one.
- **Pros**: End-to-end typesafety; auth, ORM, API layer all wired.
- **Cons**: Same rendering model as Pick A (so this is a *batteries*
  contrast, not a *philosophy* contrast); replacing any one piece is
  friction.

---

## 4_forum — auth + threaded posts + CRUD

### Pick A — SvelteKit
```bash
npx sv create forum
```
- **Why**: Form actions + load functions feel like classic server-rendered
  forums but with a modern reactive UI. Very small bundles, one-command
  scaffold — fits "lightweight" tightly.
- **Pros**: Tiny payload (great for first post-load), built-in form
  handling and CSRF, simple mental model.
- **Cons**: Smaller ecosystem (fewer prebuilt rich-text editors, auth
  recipes) than React/Next.

### Pick B — Django + HTMX
```bash
python -m venv venv && source venv/bin/activate
pip install "django>=5" django-htmx
django-admin startproject forum
```
- **Why**: Forums are the canonical Django use case (admin panel, auth,
  ORM, moderation tooling, permissions). HTMX makes the UI feel modern
  without an SPA. Heavier *first-touch* than SvelteKit, but the most
  "natural" solution for the problem shape.
- **Pros**: Most production-grade option for moderation, admin,
  permissions; tiny client-side JS; mature batteries-included story.
- **Cons**: Two languages (Python + a sprinkle of JS); deployment is
  heavier (WSGI/ASGI server, static collection); not a one-liner CLI.

### Pick C — T3 stack (Next + tRPC + Prisma + NextAuth)
```bash
npm create t3-app@latest
```
- **Why**: If the forum will eventually have a heavy SPA-ish UI (live
  voting, embedded media, web push), T3's typed Next stack is a natural
  destination.
- **Pros**: One typed schema from DB → API → UI; NextAuth covers OAuth
  and email magic links.
- **Cons**: Heaviest of the three; tRPC is awkward if you later need a
  public REST API.

---

## 5_travel-booking — marketing-heavy + light interactivity

### Pick A — Astro + Tailwind
```bash
npm create astro@latest
```
- **Why**: 90% of these pages are content (Home, About, Packages, Tour
  Plan, Gallery). Only "Build Your Own Package" is interactive.
- **Pros**: Near-zero JS, fast image-heavy galleries, great SEO; the
  package builder can be a single React/Vue/Svelte island.
- **Cons**: Booking flow with payments needs an external service (Stripe
  Checkout, Snipcart) or a separate backend.

### Pick B — Next.js
```bash
npx create-next-app@latest --ts --tailwind --eslint --app
```
- **Why**: One stack for marketing pages and the package-builder/booking
  flow; ISR for package pages so they stay snappy.
- **Pros**: Server Actions are great for booking submission; image and
  font optimization built-in.
- **Cons**: Overkill if you never wire real bookings; ships more JS than
  Astro for what's mostly content.

### Pick C — Eleventy + Alpine.js
```bash
npm init -y && npm install --save-dev @11ty/eleventy
# add Alpine via <script> in your layout
```
- **Why**: Cheapest, fastest possible delivery for a brochure site;
  Alpine.js gives the package-builder interactivity in ~10KB.
- **Pros**: Smallest possible footprint; deploys to any static host;
  Markdown for tour content.
- **Cons**: No component model means duplication across pages; auth/
  booking require an external backend.

---

## 6_chat — realtime messaging + voice/video

### Pick A — SvelteKit + Supabase Realtime
```bash
npx sv create chat
npm i @supabase/supabase-js
```
- **Why**: SvelteKit is light on the wire and Supabase Realtime
  (Postgres → websocket) covers messages, presence, and auth out of the
  box — the most practical "lightweight chat starter" path. Voice/video
  pluggable via Daily/LiveKit.
- **Pros**: Tiny bundles, one-command scaffold, generous free tier; you
  only write your UI; auth + realtime + DB are a single dependency.
- **Cons**: Tied to Supabase's realtime semantics; very heavy rooms
  (10k+ users in one channel) will need a different transport.

### Pick B — Phoenix LiveView (Elixir)
```bash
mix phx.new chat --live
```
- **Why**: Technically the cleanest solution for many concurrent
  connections, presence, and pub/sub — what production chat apps end up
  rebuilding by hand.
- **Pros**: Best-in-class for high-concurrency realtime; very low client
  JS; presence built in.
- **Cons**: Elixir is a new language for most teams (highest learning
  curve of the three); voice/video still needs WebRTC + a TURN server
  (LiveKit, Twilio, Daily).

### Pick C — Next.js + Socket.IO (or tRPC subscriptions)
```bash
npx create-next-app@latest --ts --tailwind --eslint --app
npm i socket.io socket.io-client
```
- **Why**: Familiar React/Node stack; Socket.IO is well-trodden for chat;
  voice/video via the LiveKit React SDK.
- **Pros**: Huge ecosystem; pre-built UI kits (stream-chat-react,
  Sendbird).
- **Cons**: Socket.IO doesn't fit serverless cleanly — you'll likely run
  a long-lived Node server, which complicates Vercel-style deploys.

---

## 7_cloud-storage — product front-end + heavy admin / SaaS dashboard

### Pick A — Next.js (App Router) + shadcn/ui
```bash
npx create-next-app@latest --ts --tailwind --eslint --app
npx shadcn@latest init
```
- **Why**: This task is *both* a product front-end (welcome, file
  browser, photo grid, transfer progress, billing) *and* an admin panel
  (users, plans, transactions, payments). Next + shadcn is the most
  balanced foundation for that hybrid — neither half feels bolted on.
- **Pros**: Total control; modern stack; easy to layer auth (Clerk,
  NextAuth); shadcn ships sane defaults for tables/forms/charts the
  admin half needs.
- **Cons**: You build the admin patterns yourself — slower than Refine
  for the dozens of CRUD screens.

### Pick B — Refine (React, headless admin framework)
```bash
npm create refine-app@latest
```
- **Why**: The admin half (Users, Plans, Transactions, Payments,
  Branding, Pages, Languages…) is exactly what Refine generates from a
  data source. Best fit *if* you treat the file-browser part as a
  separate app or accept that it'll be hand-built.
- **Pros**: Massive head start on admin screens; works with REST,
  GraphQL, Supabase, Strapi; pluggable UI (Ant/MUI/Chakra/shadcn).
- **Cons**: Opinionated routing/data layer; the file browser, photo
  grid, and welcome/landing pages feel less natural than in a plain
  Next app.

### Pick C — Vite + React + React Admin
```bash
npm create vite@latest cloud-storage -- --template react-ts
npm i react-admin ra-data-simple-rest
```
- **Why**: Pure SPA + React Admin gives you the fastest dev speed for
  the admin half; pair with any backend (Strapi, Hasura, Supabase,
  custom REST).
- **Pros**: Extremely fast for CRUD screens; tons of built-in components.
- **Cons**: Public pages and SEO are weak (it's a SPA); React Admin's
  default UI looks "internal-tool-y" without theming work.

---

## 8_ecommerce — sign-in/up, shop, product, cart, checkout (7 pages)

### Pick A — Next.js + Stripe Checkout (lightweight storefront)
```bash
npx create-next-app@latest --ts --tailwind --eslint --app
npm i stripe @stripe/stripe-js
```
- **Why**: For a 7-page mockup, you don't need a full commerce backend.
  Plain Next gives you the storefront UI; Stripe Checkout (hosted) gives
  you cart-to-payment without writing payment code.
- **Pros**: Simplest possible storefront stack; SSR/ISR for product
  pages; complete payment flow in ~50 lines; no provider coupling.
- **Cons**: Hosted Stripe Checkout limits cart UX customization; you
  still need *some* product/order data store (a JSON file, Supabase,
  or Sanity).

### Pick B — Next.js Commerce (Vercel reference storefront)
```bash
npx create-next-app@latest -e https://github.com/vercel/commerce
```
- **Why**: Reference storefront with product/cart/checkout already wired
  to a pluggable commerce provider (Shopify, BigCommerce, Medusa,
  Saleor). The right pick if you have, or plan to have, a real backend.
- **Pros**: Production-grade out of the box; great performance and SEO;
  swap commerce backends without rewriting UI.
- **Cons**: Tightly coupled to a "commerce provider" — heavy for a
  benchmark; if you want a fully custom backend, you're either using
  Medusa or rewriting adapters.

### Pick C — Astro + Snipcart (or Stripe Checkout)
```bash
npm create astro@latest
# add Snipcart via <script> + data attributes, or Stripe Checkout via API
```
- **Why**: Simplest possible storefront — fast static product pages,
  cart and payments handled by an external service.
- **Pros**: Cheapest to host; fastest pages; minimal backend code.
- **Cons**: Limited control over checkout UX; ongoing fees on every
  sale; not a fit if you need customer accounts/history.

> Heavier alternative worth knowing: **Medusa.js**
> (`npx create-medusa-app@latest`) — open-source headless commerce
> backend + a Next storefront. Right answer if you outgrow Pick A and
> want to *own* the backend (products, orders, fulfillment, promotions)
> without writing it yourself. Excluded from the top 3 because it's
> overkill for a 7-page benchmark.

---

## 9_project-management — interactive dashboard (Kanban, Worklog, Reports)

### Pick A — Vite + React + TS (pure SPA)
```bash
npm create vite@latest pm-app -- --template react-ts
```
- **Why**: This is an authenticated app behind a login — SEO doesn't
  matter, but interactivity does. A SPA + a typed API (tRPC, REST,
  GraphQL) is the simplest model.
- **Pros**: Fast HMR, tiny config, free choice of state library
  (Zustand/Redux Toolkit); best fit for drag-and-drop Kanban; persistent
  layouts are trivial.
- **Cons**: You bring your own router (TanStack Router/React Router),
  auth, and data layer; no SSR.

### Pick B — Next.js (App Router) + Server Actions
```bash
npx create-next-app@latest --ts --tailwind --eslint --app
```
- **Why**: Mix dashboard interactivity with server-rendered data and
  Server Actions for mutations (move card, log work, comment).
- **Pros**: Less plumbing than a SPA + separate API; built-in auth
  patterns (NextAuth/Clerk); good for shareable report URLs.
- **Cons**: Drag-and-drop and complex client state still need careful
  Client Component boundaries; extra ceremony vs a pure SPA.

### Pick C — SvelteKit
```bash
npx sv create pm-app
```
- **Why**: Reactive store model is a great fit for Kanban/board UIs;
  smaller bundle than React for the same UX.
- **Pros**: `$state`/stores make drag-and-drop and live updates feel
  natural; load functions handle SSR data fetching cleanly.
- **Cons**: Fewer prebuilt UI kits (calendars, gantt, kanban) than the
  React ecosystem; smaller hiring pool.

---

## 10_streaming_music-streaming — library + persistent player

### Pick A — Vite + React + TS (SPA)
```bash
npm create vite@latest music -- --template react-ts
```
- **Why**: Streaming apps are highly interactive and authenticated; a
  SPA shell keeps the `<audio>` element mounted forever and avoids
  full re-renders on navigation. SSR adds nothing for the player itself.
- **Pros**: Simplest model for global audio state, queue, and shuffle;
  fast iteration; no risk of accidentally unmounting playback on
  route changes.
- **Cons**: Worse SEO and slower first paint for browse/discover pages
  than SSR; you build your own routing/auth.

### Pick B — Next.js (App Router)
```bash
npx create-next-app@latest --ts --tailwind --eslint --app
```
- **Why**: Library/browse/premium pages benefit from SSR + caching;
  parallel/intercepting routes can host a persistent player above the
  routed content.
- **Pros**: Spotify-style "persistent player while routes change" is
  doable with parallel routes; good for the marketing/premium-upgrade
  pages alongside the app.
- **Cons**: Persistent audio across route transitions takes care
  (don't unmount the `<audio>` element); SSR adds nothing for the
  player itself.

### Pick C — Nuxt 3 (Vue)
```bash
npx nuxi@latest init music
```
- **Why**: Vue's `<Teleport>` + persistent layouts make a "player at
  the bottom that survives route changes" especially clean.
- **Pros**: Friendly DX, auto-imports, hybrid SSR/SPA per route.
- **Cons**: Smaller audio/streaming-specific component ecosystem than
  React; fewer turn-key integrations.

---

## TL;DR — quick matrix (v2)

| Task                          | Pick A (recommended)            | Pick B                       | Pick C                        |
|-------------------------------|---------------------------------|------------------------------|-------------------------------|
| 1_newsletter                  | Astro                           | Eleventy                     | Next.js + MDX                 |
| 2_real-estate                 | Next.js                         | Nuxt 3                       | Astro + islands               |
| 3_job-board                   | Next.js                         | **Remix** ⬆                  | T3 stack ⬇                    |
| 4_forum                       | SvelteKit                       | **Django + HTMX** ⬆          | T3 stack ⬇                    |
| 5_travel-booking              | Astro                           | Next.js                      | Eleventy + Alpine.js          |
| 6_chat                        | **SvelteKit + Supabase** ⬆      | Phoenix LiveView ⬇           | Next.js + Socket.IO           |
| 7_cloud-storage               | **Next.js + shadcn/ui** ⬆       | Refine ⬇                     | Vite + React Admin            |
| 8_ecommerce                   | **Next.js + Stripe Checkout** ⬆ | Next.js Commerce ⬇           | Astro + Snipcart              |
| 9_project-management          | Vite + React (SPA)              | Next.js (App Router)         | SvelteKit                     |
| 10_streaming_music-streaming  | **Vite + React (SPA)** ⬆        | Next.js ⬇                    | Nuxt 3                        |

> ⬆ promoted in v2, ⬇ demoted in v2.

---

## Selection criteria (made explicit in v2)

In v1 the heuristic was implicit and Next.js-biased. In v2 a pick has to
satisfy **all four** of these to land at A:

1. **One-command scaffold** — there is a single CLI that produces a
   runnable project (`npm create …`, `npx create-… `, `mix … new`,
   `django-admin startproject`).
2. **Small runtime footprint** — small JS bundle, or a server-rendered
   stack with minimal client JS.
3. **Natural fit for the task's *primary* axis** — content vs CRUD vs
   realtime vs admin vs interactive-app. The A pick must not need a
   second framework to cover the primary axis.
4. **Genuinely different philosophy from B and C** — A/B/C should not
   collapse into "same framework, different opinions". If two picks
   share a rendering model, they should at least differ in data layer
   or batteries-included-ness.
