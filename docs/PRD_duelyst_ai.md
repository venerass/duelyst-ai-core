# DUELYST.AI — Product Requirements Document

> **A Platform Where AIs Debate the Big Questions**
>
> Prepared for: Veneraldo
> April 2026 | Version 1.0
> PRD + Technical Architecture + Implementation Guide

---

## Table of Contents

1. [Product Vision & Strategy](#product-vision--strategy)
2. [Product Requirements](#product-requirements)
3. [Technical Architecture](#technical-architecture)
4. [Growth & Content Strategy](#growth--content-strategy)
5. [Development Phases](#development-phases)
6. [Financial Projections](#financial-projections)
7. [Risks & Mitigations](#risks--mitigations)
8. [Success Metrics & Decision Points](#success-metrics--decision-points)

---

## Product Vision & Strategy

### Overview

Duelyst.ai is a platform where AI models debate topics, questions, and decisions in structured multi-turn exchanges. Two AI agents, each powered by a model chosen by the user (Claude, GPT, Gemini), research, argue, challenge each other, and converge toward a synthesis. Every debate can be published as shareable content with a unique URL, creating an ever-growing library of AI-generated analysis on every subject imaginable.

The platform combines three value propositions into a single product:

- **Content platform** — AI debates generate engaging, indexable, shareable pages.
- **Decision tool** — two adversarial AIs produce more reliable conclusions than asking one model alone.
- **Media engine** — transforms debates into podcasts, summaries, and visual reports.

### Problem Statement

Current AI interactions suffer from two well-documented failure modes:

1. **Sycophancy** — a single AI model tends to agree with the user rather than challenge their assumptions, producing answers that feel satisfying but may be incomplete or biased.
2. **Hallucination** — a single model can state fabricated facts with high confidence, and the user has no built-in mechanism to detect this.

When two AI models are placed in adversarial dialogue, they naturally challenge each other's claims, request evidence, and surface disagreements. This structural dynamic reduces both failure modes. The result is analysis that is more thorough, more balanced, and more trustworthy than what any single model produces alone.

Beyond the quality improvement, there is currently no platform where users can create, consume, and share AI-generated debates as content. This is a content category that does not exist yet.

### Target Users

**Primary: Content Creators & Power Users.** Tech enthusiasts, AI power users, bloggers, and content creators who want to generate unique, shareable content by pitting AI models against each other on trending topics. They are drawn by the novelty, the viral potential of "Claude vs GPT" headlines, and the ability to publish debates as blog-style posts.

**Secondary: Decision Makers.** Professionals (founders, product managers, investors, engineers) who face complex decisions and want structured adversarial analysis before committing. They use private debates to stress-test strategies, evaluate trade-offs, and surface blind spots.

**Tertiary: Curious Consumers.** Readers who discover published debates via search or social media. They come to read, not create. They are the audience that makes the content flywheel work. Some convert to creators over time.

### Core Value Proposition

Duelyst.ai is the platform where AIs debate the big questions. Users create structured debates between top AI models, each equipped with web search, data analysis, and code execution. Every debate becomes shareable content. The result is analysis that no single AI can produce alone.

### Competitive Landscape

No direct competitor offers adversarial multi-agent debates with tool use, published as shareable content. The closest alternatives each address only a fragment of the value proposition:

| Competitor | What They Do | What They Miss |
|---|---|---|
| ChatGPT / Claude / Gemini | Single-model Q&A | No adversarial challenge, no published debates |
| ChatGPT Arena (LMSYS) | Blind model comparison via voting | No structured debate, no tools, no publishing |
| Perplexity | Search-augmented single-model answers | No adversarial dialogue, no content platform |
| Multi-tab manual | User opens 2 tabs, copies between them | Tedious, no tool use, no synthesis, no sharing |
| Consensus.app | Academic paper synthesis | Limited to papers, no AI debate, no user topics |

### Moat Strategy

The core engine (LangGraph agents calling LLM APIs) is technically replicable. The moat is built on three layers that compound over time:

1. **Accumulated content.** Every published debate is an indexable page. At 10,000+ debates covering technology, politics, science, finance, and culture, the library becomes a reference asset that no competitor can replicate quickly. Each page is a long-tail SEO entry point.
2. **Brand as category.** If "Duelyst" becomes synonymous with "AI debates" in the way "Uber" became synonymous with ride-sharing, that recognition is moat. The open-source core reinforces this: every developer using `duelyst-ai-core` is spreading the brand.
3. **Community of creators.** Creators who build audiences on the platform (followers, popular debates, reputation) do not migrate easily. Network effects strengthen as more creators attract more readers.

---

## Product Requirements

### User Flows

#### Flow 1: Create a Debate

1. **Define topic.** User enters a question or topic (e.g., "Should I learn Rust or Go in 2026?").
2. **Configure agents.** User selects a model for each agent (Claude, GPT, Gemini). Optionally assigns a side or specific instructions (e.g., "Agent A: defend Rust. Agent B: defend Go.").
3. **Launch debate.** Debate runs in real-time. User watches the turns stream in, seeing each agent research, argue, and respond.
4. **Receive synthesis.** A neutral judge agent (third model, different from the debaters) produces a final synthesis: key arguments from each side, points of agreement, points of disagreement, and a conclusion.
5. **Publish or save.** User publishes to the platform (public URL, SEO-optimized page) or saves as private.

#### Flow 2: Consume Published Debates

1. **Discover.** User arrives via search engine, social media link, or the platform homepage (trending debates, featured, categories).
2. **Read.** Debate page shows: title, models used, each turn as a structured conversation, evidence and visualizations generated during the debate, and the final synthesis.
3. **Interact.** User can upvote, share (Twitter/X, LinkedIn, copy link), comment, or create a follow-up debate on the same topic.
4. **Convert.** Reader creates an account and launches their own debate, entering the creator flywheel.

#### Flow 3: Generate Podcast

1. **Select debate.** From a completed debate, user clicks "Generate Podcast".
2. **Processing.** Backend generates audio: each agent gets a distinct voice (via TTS API), with intro/outro and natural conversational flow.
3. **Deliver.** User receives an MP3 for download, or a shareable podcast page with embedded player.

### Feature Requirements by Tier

| Feature | Free | Pro ($9.99/mo) |
|---|---|---|
| Debates per week | 5 | 20 |
| Private debates per week | 2 | 20 |
| Models available | GPT-5.2-mini, Haiku, Flash | All (Opus, GPT-5.4, Pro) |
| Rounds per debate | 3 | Up to 10 |
| Web search tool | Yes | Yes |
| Code execution tool | Yes | Yes |
| Data visualization | Yes | Yes |
| Podcast generation | No | Yes (5/month) |
| Published debate page | Yes (after the 2nd) | Yes (optional) |
| Custom agent instructions | Basic | Advanced |
| Export (PDF, Markdown) | No | Yes |
| API access | No | No |

> **Key design decision:** Free tier debates are public by default. Every free debate published feeds the content flywheel (SEO, social sharing, library growth). Users get 2 private debates per week to experiment without publishing low-quality content. This aligns free usage with platform growth.

### Published Debate Page Requirements

Each published debate is a standalone web page optimized for consumption and sharing:

- **SEO-optimized.** Title tag, meta description, Open Graph image (auto-generated), canonical URL, `schema.org` markup for Article type.
- **Clean URL structure.** `duelyst.ai/debates/[slug]` where slug is derived from the topic (e.g., `duelyst.ai/debates/rust-vs-go-2026`).
- **Rich content.** Each turn displayed as a structured card showing the agent name, model used, argument text, evidence citations, and any generated visualizations or code outputs.
- **Synthesis section.** Final synthesis prominently displayed with key takeaways, agreement points, and conclusion.
- **Social sharing.** Twitter/X card with auto-generated preview image (models used + topic + key conclusion). LinkedIn, Reddit, and copy-link buttons.
- **Engagement.** Upvote count, view count, share count. Comment section (authenticated users only).
- **Related debates.** Links to debates on similar topics to increase time-on-site and discovery.

### Agent Capabilities (Technical Requirements)

Each agent in a debate is a LangGraph graph with the following capabilities:

| Capability | Description | Implementation |
|---|---|---|
| Multi-model support | Claude (Anthropic), GPT (OpenAI), Gemini (Google) | LangChain model adapters, switchable per agent |
| Web search | Search the internet for current data and evidence | Tavily API (primary), SerpAPI (fallback) |
| Code execution | Run Python code to generate calculations and charts | E2B sandbox (production), local Docker (OSS) |
| Data visualization | Generate charts and graphs as evidence | Matplotlib/Plotly in sandboxed execution |
| Reflection | Analyze opponent arguments before responding | ReAct-style reasoning step in agent graph |
| Convergence detection | Signal agreement or persistent disagreement | Structured output with convergence score per turn |
| Citation tracking | Reference sources found via web search | Metadata attached to each claim in agent output |
| User instructions | Follow user-defined side or constraints | Injected into agent system prompt per debate config |

### Debate Orchestration Logic

The orchestrator manages the debate flow and decides when to terminate:

- **Turn structure.** Each round consists of Agent A responding, then Agent B responding. The orchestrator passes the full debate history to each agent on their turn.
- **Convergence scoring.** Each agent outputs a convergence score (0–10) indicating how much they agree with the opponent. When both agents score above 7 for two consecutive rounds, the debate enters synthesis phase.
- **Maximum rounds.** Hard cap based on user tier (3 for free, up to 10 for Pro). If max rounds reached without convergence, synthesis still runs, noting the persistent disagreement.
- **Synthesis generation.** A third agent (the judge) receives the full debate transcript and produces: summary of each side, key evidence cited, points of agreement, points of disagreement, and a balanced conclusion. The judge model is always different from the debaters to avoid bias.
- **Streaming.** Each agent turn is streamed to the frontend via SSE (Server-Sent Events) as it generates, so the user watches the debate unfold in real-time.

---

## Technical Architecture

### Architecture Pattern: Centralized (Pattern 2)

Based on the Architecture Guide for Solo AI Projects, Duelyst.ai uses **Pattern 2 (Centralized)** where the frontend never talks directly to the database. All operations pass through the Python backend, which is the single point of authentication, authorization, and business logic.

**Rationale:** The LangGraph debate engine is the core of the product and requires Python. Debates involve long-running processes (2–5 minutes with web search and code execution) that need persistent connections and streaming. Data includes user-generated content that will be publicly published, requiring careful moderation and validation. The centralized pattern provides a single security surface for auditing.

### System Architecture Diagram

```
User → Frontend (Next.js on Vercel)
         └── Backend Python (FastAPI on Railway) ─ ALL logic
                  ├── Auth: validates Supabase JWT
                  ├── Debate Engine: LangGraph (imports duelyst-ai-core)
                  ├── Database: Supabase PostgreSQL (service role)
                  ├── Storage: Supabase Storage (podcasts, images)
                  ├── LLM APIs: Anthropic, OpenAI, Google
                  ├── Search: Tavily API
                  ├── Code Sandbox: E2B
                  └── TTS: ElevenLabs / OpenAI TTS
```

> **Exception:** Authentication (login/signup) uses Supabase Auth SDK on the frontend for OAuth flows (Google Sign-In, GitHub). The frontend obtains a JWT and sends it to the backend with every request.

### Repository Strategy: Open-Core

The project follows an open-core model with three repositories. The open-source core attracts community, stars, and brand awareness. The proprietary API and frontend contain the product logic, monetization, and user experience.

#### Repository 1: `duelyst-ai-core` (Public, OSS)

| Attribute | Detail |
|---|---|
| Repository | `github.com/venerass/duelyst-ai-core` |
| Visibility | Public (MIT license) |
| Language | Python |
| Distribution | PyPI (`pip install duelyst-ai-core`) |
| Purpose | LangGraph debate engine, model adapters, tool integrations, CLI |

This repo contains the debate engine as a standalone Python package. A developer can install it, provide their own API keys, and run debates from the terminal or integrate into their own projects. Contents:

- LangGraph agent definition (reflection, research, argument, convergence scoring)
- Model adapters: Claude (Anthropic SDK), GPT (OpenAI SDK), Gemini (Google GenAI SDK)
- Tool integrations: web search (Tavily), code execution (E2B / local Docker)
- Orchestrator: turn management, convergence detection, synthesis generation
- CLI: `duelyst debate --topic "..." --model-a claude --model-b gpt`
- Output formatters: Markdown, JSON, terminal rich output

#### Repository 2: `duelyst-ai-api` (Private)

| Attribute | Detail |
|---|---|
| Repository | `github.com/venerass/duelyst-ai-api` (private) |
| Language | Python (FastAPI) |
| Deployment | Railway (Docker container, always-on) |
| Purpose | Product backend: auth, persistence, streaming, billing, moderation |

This repo imports `duelyst-ai-core` as a pip dependency and wraps it with product logic:

- Authentication middleware (Supabase JWT validation)
- Debate CRUD: create, retrieve, list, delete debates in Supabase
- SSE streaming endpoint for real-time debate viewing
- Publication system: slug generation, SEO metadata, Open Graph image generation
- Podcast generation queue (background task via asyncio or Celery)
- Rate limiting and billing enforcement per user tier
- Content moderation (pre-publish validation)
- Admin endpoints for featured debates and editorial curation

#### Repository 3: `duelyst-ai-app` (Private)

| Attribute | Detail |
|---|---|
| Repository | `github.com/venerass/duelyst-ai-app` (private) |
| Language | TypeScript (Next.js) |
| Deployment | Vercel |
| Purpose | Frontend: debate creator, real-time viewer, published pages, user profiles |

The frontend is a Next.js application with two distinct modes:

- **Application mode.** Authenticated pages: debate creator, dashboard, settings, private debates. Client-side rendering with API calls to the backend.
- **Content mode.** Public debate pages: server-side rendered (SSR) or statically generated (ISR) for SEO. Homepage with trending debates, category pages, individual debate pages. These pages are the primary SEO entry points.

### Dependency Flow

```
duelyst-ai-core (PyPI package)
        │
        ├── pip install duelyst-ai-core
        │
        ▼
duelyst-ai-api (FastAPI on Railway)
        │
        ├── HTTPS + SSE
        │
        ▼
duelyst-ai-app (Next.js on Vercel)
```

The dependency is strictly unidirectional: the API imports the core, the app calls the API. The core never knows about the API or the app. This ensures the OSS package remains clean and product-independent.

### Infrastructure & Services

#### Service Map

| Component | Service | Plan | Monthly Cost (MVP) | Purpose |
|---|---|---|---|---|
| Database + Auth | Supabase | Free → Pro ($25) | $0–25 | PostgreSQL, Auth (JWT), Storage (podcasts/images) |
| Backend hosting | Railway | Hobby ($5 + usage) | $5–15 | FastAPI Docker container, always-on, auto-deploy |
| Frontend hosting | Vercel | Hobby (free) | $0 | Next.js, CDN, ISR for debate pages, preview deploys |
| LLM APIs | Anthropic + OpenAI + Google | Pay-per-use | $20–100 | Model inference for debates |
| Web search | Tavily | Free → Growth | $0–50 | Agent web search tool (1000 free/month) |
| Code sandbox | E2B | Free → Pro | $0–30 | Sandboxed Python execution for agent tools |
| TTS (podcasts) | ElevenLabs or OpenAI TTS | Pay-per-use | $5–20 | Voice generation for podcast feature |
| Error tracking | Sentry | Free | $0 | Error monitoring for backend and frontend |
| Uptime | BetterStack | Free | $0 | Uptime monitoring and alerting |
| Domain + DNS | Cloudflare | Free | $1 (domain) | DNS, CDN edge, SSL |
| Source control | GitHub | Free (public) + Pro | $4 | Repositories, CI/CD, Dependabot |

**Total estimated MVP cost: $35–80/month.** The dominant cost is LLM API usage, not infrastructure. At scale, LLM costs grow linearly with debates while infrastructure costs remain relatively flat.

### Why Supabase (Not Firebase)

- **Relational data model fits naturally.** Debates have structured relationships: users → debates → turns → evidence. PostgreSQL handles complex queries (trending debates, filtered searches, aggregations) natively. Firestore requires denormalization and composite indexes for every query pattern.
- **Published debates need complex queries.** "Top debates this week" requires ordering by a composite of upvotes, views, and recency. "Debates about Bitcoin using Claude" requires multi-field filtering. PostgreSQL does this in one query; Firestore requires pre-computed indexes or Cloud Functions for each pattern.
- **Full-text search.** Supabase includes PostgreSQL full-text search, enabling search across debate topics and content without an external service.
- **Integrated storage.** Supabase Storage (S3-compatible) handles podcast MP3 files and generated images with the same auth system. No separate Firebase Storage configuration needed.
- **JWT-based auth integrates cleanly with FastAPI.** Supabase Auth issues standard JWTs that the Python backend validates directly. Firebase requires the Firebase Admin SDK, which is heavier.

### Why Railway for Backend

- **Always-on container.** No cold start. Debates start immediately when a user creates one. Railway runs the Docker container continuously.
- **Native streaming support.** SSE (Server-Sent Events) connections stay open for minutes during a debate. Railway supports long-lived HTTP connections without timeouts.
- **GitHub integration.** Push to main triggers automatic redeploy. Preview environments for PRs.
- **Simple scaling.** Increase container resources (RAM, CPU) with a slider. Add replicas when needed. No Kubernetes configuration.
- **Cost-efficient at MVP scale.** $5/month base plus usage. A single container handles hundreds of concurrent debates before needing scaling.

### Database Schema (Core Tables)

| Table | Key Columns | Purpose |
|---|---|---|
| `users` | id, email, display_name, plan, created_at | User accounts (synced from Supabase Auth) |
| `debates` | id, user_id, topic, slug, status, config, model_a, model_b, instructions_a, instructions_b, is_public, published_at, views, upvotes | Debate metadata and configuration |
| `turns` | id, debate_id, agent (a/b/judge), turn_number, content, evidence[], convergence_score, model_used, tokens_used, created_at | Individual debate turns with full content |
| `publications` | id, debate_id, slug, title, description, og_image_url, seo_metadata, featured, category | Published debate pages (SEO data, editorial flags) |
| `podcasts` | id, debate_id, audio_url, duration, status, created_at | Generated podcast files |
| `votes` | id, debate_id, user_id, created_at | Upvotes on published debates |
| `comments` | id, debate_id, user_id, content, created_at | Comments on published debates |

### Security Architecture

#### Authentication & Authorization

- **Supabase Auth on frontend.** Handles OAuth flows (Google, GitHub) and issues JWTs. Frontend stores the JWT and sends it with every API request.
- **JWT validation middleware on backend.** Every FastAPI endpoint validates the JWT signature, expiration, and issuer. The `user_id` is always extracted from the token, never from the request body.
- **Tier-based authorization.** Backend checks user plan (free/pro/teams) before allowing access to premium features (model selection, tools, podcast generation).
- **RLS as defense in depth.** Even though the backend uses service role, Supabase RLS policies are configured as a second safety layer. If the Python code has a bug, RLS prevents data leakage.

#### Prompt Injection Protection

- **User input (topic, instructions) is never injected into system prompts.** System prompts are static templates. User content is passed as clearly delimited user messages with XML tags.
- **Agent outputs are validated before storage and display.** HTML is sanitized (DOMPurify on frontend, bleach on backend). Code execution outputs are sandboxed in E2B.
- **Debate topics and instructions are validated** for length (max 2000 chars), format, and basic content moderation before being sent to agents.

#### API Protection

- **Rate limiting.** Per-user rate limits on debate creation (based on tier). Per-IP rate limiting on all endpoints. Specific strict limits on LLM-invoking endpoints to prevent cost explosion.
- **CORS.** Restricted to `duelyst.ai` domain only. No wildcard origins.
- **Request validation.** Pydantic models on every endpoint. Strict input schemas. No unvalidated user input reaches the database or LLM APIs.
- **Cost controls.** Spending limits configured on Anthropic, OpenAI, and Google accounts. Daily monitoring alerts for usage spikes above 2x average.

#### Secret Management

- All API keys (Anthropic, OpenAI, Google, Tavily, E2B, ElevenLabs, Supabase service role) stored as Railway environment variables. Zero secrets in code.
- Frontend only has Supabase anon key (public) and API base URL. No LLM keys, no service role key.
- `gitleaks` configured in CI pipeline on both private repos. Blocks merge if secrets detected.
- Key rotation every 90 days for all API keys.

---

## Growth & Content Strategy

### The Content Flywheel

The growth model is a self-reinforcing content flywheel:

1. **Debates create content.** Every published debate is a page with a unique URL, title, and SEO metadata.
2. **Content drives traffic.** Published debates rank in search engines and are shared on social media. Each page is a long-tail SEO entry point.
3. **Traffic creates readers.** Visitors discover the platform through published debates. They consume content without an account.
4. **Readers become creators.** Some readers sign up and create their own debates, adding more content to the platform.
5. **Creators attract readers.** More content attracts more search traffic and social shares. The flywheel accelerates.

### Free Tier as Growth Engine

The free tier is designed to maximize content creation:

- All free debates are public by default. Each free user is creating SEO-indexable content for the platform.
- 2 private debates per week lets users experiment without publishing low-quality content.
- Free tier uses cheaper models (GPT-5.2-mini, Haiku, Flash), reducing cost per debate to approximately R$0.50–1.00 while still producing quality content.
- The upgrade path is clear: users who want privacy, frontier models, and podcast generation upgrade to Pro.

### Editorial Content (Seed Strategy)

Before the user base generates enough content, Veneraldo creates editorial debates weekly:

- **Weekly trending debate.** A curated debate on a trending topic (e.g., "Claude vs GPT: Will AI replace software engineers by 2030?"). Published on the platform and shared on Twitter/X, Reddit (r/artificial, r/ChatGPT, r/technology), Hacker News.
- **Category seeding.** 5–10 high-quality debates per category (technology, finance, science, politics, philosophy) to populate the platform before launch.
- **Format experimentation.** Test different debate formats (prediction, comparison, devil's advocate, analysis) to learn which generate the most engagement.

### Distribution Channels

| Channel | Strategy | Timing |
|---|---|---|
| Hacker News | Launch the OSS core (`duelyst-ai-core`) as a Show HN. Technical audience, high-quality feedback. | Phase 1 (OSS launch) |
| Product Hunt | Launch the platform with curated featured debates. Category: AI / Productivity. | Phase 3 (platform launch) |
| Reddit | Share debates in relevant subreddits. r/artificial, r/ChatGPT, r/technology, topic-specific subs. | Ongoing from Phase 2 |
| Twitter/X | Share debate highlights, "Claude vs GPT" clips, podcast snippets. Build in public. | Ongoing from Phase 1 |
| SEO | Published debates as long-tail pages. Blog with AI analysis content. | Compounds from Phase 3 |
| OSS community | `duelyst-ai-core` stars, forks, contributors. Each user reinforces the brand. | Phase 1 onward |

---

## Development Phases

### Phase 1: OSS Engine + CLI (Weeks 1–3) ✅

**Goal:** Ship the `duelyst-ai-core` package to PyPI and GitHub. Validate the debate engine technically and generate initial buzz in the developer community.

**Deliverables:**

- LangGraph agent with support for Claude, GPT, and Gemini
- Orchestrator with turn management and convergence detection
- Web search tool integration (Tavily)
- CLI: `duelyst debate --topic "..." --model-a claude --model-b gpt`
- Markdown and JSON output formatters
- README with examples and installation instructions
- Published on PyPI

**Launch:** Post on Hacker News (Show HN), Reddit r/LangChain, r/artificial. Share on Twitter/X.

**Success metric:** 100+ GitHub stars, 500+ PyPI installs in first 2 weeks.

### Phase 2: API + Frontend MVP (Weeks 4–7)

**Goal:** Ship the web platform where users can create, watch, and share debates.

**Deliverables:**

- FastAPI backend importing `duelyst-ai-core`, deployed on Railway
- Supabase setup: database schema, auth, storage
- SSE streaming endpoint for real-time debate viewing
- Next.js frontend: debate creator, real-time viewer, debate detail page
- Supabase Auth integration (Google + GitHub OAuth)
- Basic debate listing page (user's debates)

**Success metric:** 50+ debates created by beta users. Core loop (create → watch → view result) working end-to-end.

### Phase 3: Publication + SEO (Weeks 8–10)

**Goal:** Transform debates into public content pages. Launch the content flywheel.

**Deliverables:**

- Published debate pages with SSR/ISR (SEO-optimized, Open Graph, `schema.org`)
- Homepage with trending, featured, and recent debates
- Category pages (technology, finance, science, etc.)
- Social sharing (Twitter card, LinkedIn, copy link)
- Upvoting and view counting
- 20+ editorial seed debates across categories
- `sitemap.xml` and `robots.txt` for search engine indexing

**Launch:** Product Hunt launch. Reddit and Twitter/X campaign with featured debates.

**Success metric:** 500+ published debates. 10,000+ organic page views in first month. First debates appearing in Google search results.

### Phase 4: Advanced Tools + Podcast (Weeks 11–14)

**Goal:** Add the differentiating features that justify Pro pricing.

**Deliverables:**

- Code execution tool (E2B sandbox) for agents
- Data visualization generation (charts, graphs as debate evidence)
- Podcast generation from completed debates (TTS with distinct voices)
- Export to PDF and Markdown
- Comments on published debates
- User profiles with debate history

**Success metric:** Pro conversion rate above 2%. 50+ podcasts generated. Users sharing podcasts on social media.

### Phase 5: Monetization + Scale (Weeks 15+)

**Goal:** Launch paid tiers and build toward sustainable revenue.

**Deliverables:**

- Stripe integration for Pro and Teams plans
- Tier enforcement (model access, debate limits, tools, private debates)
- Teams features (shared debates, team dashboard)
- API access for Teams tier
- Embed widget (debates embeddable on external sites)

**Revenue target:** R$3,000–5,000/month by month 12 (60–100 Pro subscribers at ~$10/month).

### Timeline Summary

| Phase | Duration | Focus | Key Milestone |
|---|---|---|---|
| Phase 1: OSS Engine | Weeks 1–3 | `duelyst-ai-core` on PyPI + GitHub | 100+ stars, HN launch |
| Phase 2: API + Frontend | Weeks 4–7 | Web platform MVP | 50+ debates, core loop working |
| Phase 3: Publication | Weeks 8–10 | Public debate pages, SEO | 500+ debates, Product Hunt |
| Phase 4: Tools + Podcast | Weeks 11–14 | Code exec, viz, podcast, Pro features | 2%+ Pro conversion |
| Phase 5: Monetization | Week 15+ | Stripe, tiers, teams, embed | R$3–5k/month revenue |

---

## Financial Projections

### Cost Structure

| Cost Category | Month 1–3 | Month 4–6 | Month 7–12 |
|---|---|---|---|
| Infrastructure (Railway + Vercel + Supabase) | R$25–50 | R$75–150 | R$125–250 |
| LLM APIs (Anthropic + OpenAI + Google) | R$100–250 | R$250–750 | R$500–2,500 |
| Tavily (web search) | R$0 (free tier) | R$0–125 | R$125–375 |
| E2B (code sandbox) | R$0 (free tier) | R$0–150 | R$150–375 |
| TTS (podcast generation) | R$0 | R$25–100 | R$100–375 |
| Domain + DNS + misc | R$5 | R$5 | R$5 |
| **TOTAL** | **R$130–305/month** | **R$355–1,280/month** | **R$1,005–3,880/month** |

### Revenue Projection

| Metric | Month 6 | Month 9 | Month 12 | Month 18 |
|---|---|---|---|---|
| Published debates | 500 | 2,000 | 5,000 | 15,000 |
| Monthly visitors (organic) | 5,000 | 20,000 | 50,000 | 150,000 |
| Registered users | 500 | 2,000 | 5,000 | 15,000 |
| Pro subscribers (2–3%) | 10–15 | 40–60 | 100–150 | 300–450 |
| Monthly revenue (Pro) | R$500–750 | R$2,000–3,000 | R$5,000–7,500 | R$15,000–22,500 |
| Monthly costs | R$500–1,000 | R$800–1,500 | R$1,500–3,000 | R$3,000–5,000 |
| **Net** | **R$-250 to -500** | **R$500–1,500** | **R$2,000–4,500** | **R$10,000–17,500** |

**Path to R$15k/month:** approximately 300 Pro subscribers at R$50/month (~$10 USD). This requires roughly 10,000–15,000 registered users at a 2–3% conversion rate, which requires approximately 100,000–150,000 monthly visitors. At current SEO benchmarks for content platforms, this is achievable with 10,000–15,000 published debates. Timeline: 15–18 months.

---

## Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| LLM providers add native debate/multi-opinion features | Medium | High | Moat is in accumulated content + brand, not technology. 10,000+ debates cannot be replicated overnight. Pivot to aggregator role if needed. |
| LLM API costs make free tier unsustainable | High | Medium | Free tier uses cheapest models only. Monitor cost-per-debate closely. Reduce free limits if needed. Debates are public = marketing cost, not pure loss. |
| Low content quality drives users away | Medium | High | Editorial curation of featured debates. Quality scoring algorithm. Report/flag system. Judge synthesis quality as quality floor. |
| Slow SEO traction (content flywheel takes too long) | Medium | Medium | Supplement with Reddit, Twitter/X, Hacker News distribution. Create shareable debate highlights for social. Podcast as alternative distribution channel. |
| User-generated debates on sensitive topics cause PR issues | Medium | Medium | Content moderation pre-publish. Topic blocklist for extreme content. Clear terms of service. Report mechanism. |
| Technical complexity of multi-agent streaming | Low | Medium | Phase 1 validates the core engine before building the platform. LangGraph handles complexity. SSE is well-understood pattern. |
| Competitor launches similar platform | Low | High | First-mover advantage in content accumulation. OSS community as distribution moat. Speed of execution. |

---

## Success Metrics & Decision Points

### Month 3 Checkpoint

| Metric | Green (continue) | Red (reassess) |
|---|---|---|
| OSS stars | 200+ | Less than 50 |
| PyPI monthly downloads | 1,000+ | Less than 100 |
| Debates created on platform | 100+ | Less than 20 |
| Technical stability | Debates complete without errors >90% | Frequent failures or timeouts |

### Month 6 Checkpoint (Critical Decision)

| Metric | Green (double down) | Red (consider pivot) |
|---|---|---|
| Published debates | 500+ | Less than 100 |
| Monthly organic visitors | 5,000+ | Less than 500 |
| Registered users | 500+ | Less than 50 |
| Debates per week (user-created) | 50+ | Less than 10 |
| Social shares per week | 100+ | Less than 10 |

> **Decision at Month 6:** If metrics are red, consider pivoting effort to the AI Agent for Accounting (the Mapa de Oportunidades backup plan). Duelyst can continue as a passion project with minimal time investment while the Accounting Agent generates revenue. If metrics are green, continue investing 12–15h/week and launch monetization.

### Month 12 Checkpoint

| Metric | Green (scale) | Red (maintain/pivot) |
|---|---|---|
| Monthly revenue | R$3,000+ | Less than R$500 |
| Pro subscribers | 60+ | Less than 10 |
| Monthly organic visitors | 30,000+ | Less than 5,000 |
| Published debates | 3,000+ | Less than 500 |
| Content flywheel | User-created > editorial | Still mostly editorial |

---

*Document prepared April 2026. Review at each checkpoint.*
