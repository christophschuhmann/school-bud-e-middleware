# School Bud-E Middleware — What it is and why you might want it

**In one sentence:** this is a small “control center” that sits in front of AI services (chat, speech-to-text, text-to-speech) so your school or small business can **decide who may use which models, at what cost, and how much**—with simple web screens to manage people, pricing, and budgets.

Think of it like a front desk for AI:

* You choose which providers/models are allowed (e.g., Gemini via Google Vertex, OpenAI-compatible APIs, etc.).
* You give each user an API key and a weekly/monthly budget.
* You set prices per model (tokens, characters, or minutes).
* You decide how requests are routed (primary + fallback) if a provider is slow or unavailable.

All of this is managed from a simple browser admin page (no command line skills required).

---

## Why schools and small orgs like this

* **Privacy & control:** You pick the endpoints (including EU regions for Google Vertex, which is helpful for European schools). You can keep personal data out of the server; most frontends can store chat locally in the browser.
* **Budgets that make sense:** Every user gets a fair allowance, and there’s also an optional **Common Pool** so unused credits don’t go to waste—heavy users can borrow from it while the overall project stays inside its total cap.
* **Clear costs:** You set the prices per model (LLMs by tokens, TTS by characters, ASR by minutes/tokens). No surprises at the end of the month.
* **Simple user rollout:** Generate batches of API keys and hand them out (CSV export); you don’t need students or staff to “sign up.”

---

## What you can do in the Admin Console

Everything below happens in a clean web UI with tabs along the top.

* **Users:** add people, give them credits, generate/revoke individual API keys, filter/sort, and assign users to a project/class. 
* **Projects:** create a project (e.g., “School-A, Grade-7”), set total credits, choose a **split strategy**, and optionally turn on the **Common Pool**. There’s a **wizard** that can create a project, generate N users in one go, give everyone an allowance, and (optionally) generate their API keys. You can also **export a CSV** of users+keys and **settle the period now** to sweep unused allowance into the Common Pool immediately. 
* **Pricing:** set how much each model costs. For LLMs/VLMs you enter price per **1,000,000 tokens** (input and output). For **TTS** you enter price per **character**. For **ASR** you can bill **by tokens** (if the provider returns usage) or **by hour** (fallback). 
* **Providers:** list the upstream services you use (names, base URLs, API keys). One click can reset to sensible defaults if you want to start fresh. 
* **Routes (priority & failover):** decide which provider handles **LLM**, **TTS**, **ASR**, etc. You can set **priority** (1 = first choice) and enable/disable routes. If a top provider fails, the next one is tried automatically. Frontends can simply request “auto” and let your policy do the rest. 
* **Usage & Ledger:** see what was used, by whom, on which model, and what was billed; and see the credit changes over time. Handy for quick checks and audits. 
* **Maintenance:** download a snapshot of the database for a quick backup, restore from a snapshot, or reset everything if you’re doing a clean restart. (Simple browser upload/download—no database expertise needed.) 

---

## How budgets work (in normal words)

* Each **Project** gets a pot of credits. You can give people a recurring allowance (daily/weekly/monthly).
* If **Common Pool** is on, any **unused** allowance rolls into the pool when the period “settles.” Next period, users can draw from their allowance **plus** the pool if they need more, but the **total** still stays within the project’s overall limit. This keeps things fair but flexible. 

---

## Quick Start (non-technical)

1. **Get a small server** (e.g., a budget VPS—2 CPU cores and 2–4 GB RAM is enough).
2. **Download and start** the middleware:

   * Follow the short “QuickStart” in the repo to install requirements and run `serve.py`.
   * The first run asks you to set an **admin password** and optionally connect **Google Vertex** (you can skip this and add it later).
3. **Open the Admin page** (the script shows the local and public URL). Log in with your password.
4. In **Providers**, enter the services you use (and their API keys). In **Routes**, set who handles LLM/TTS/ASR. In **Projects**, run the wizard to make a project, users, and API keys. In **Pricing**, set your costs. You’re done. 

> We’ll publish step-by-step documentation and short YouTube tutorials so you can follow along at your own pace.

---

## What about reports and backups?

* You can export a **CSV** with users and keys for your project (useful for distribution). 
* The **Usage** and **Ledger** tabs give you a transparent view of consumption and credit changes. (PDF exports and grouped reports are supported in deployments that include the reporting add-ons.) 
* You can **download the database** from the Maintenance tab for a simple backup, and **restore** it later with a browser upload. 

---

## Where this fits

This middleware is made by **LAION** and collaborators to help schools and small organizations run AI assistants **responsibly**: you pick the models, you set the limits and prices, you keep control. It works great with the School Bud-E frontends—or any OpenAI-style client that speaks to a single endpoint.

If this sounds useful, try the Quick Start and explore the Admin Console. When you’re ready, the full documentation and tutorial videos will take you from “test it” to “roll it out.”
