# DRO Circuit Discovery Documentation

This directory contains all documentation for the DRO Circuit Discovery project. It is split into two sections serving different audiences.

## Layout

```
docs/
├── README.md                 ← You are here
├── mkdocs.yml                ← MkDocs config (renders user/ into a static site)
│
├── user/                     ← PUBLIC DOCS — for users, reviewers, collaborators
│   ├── index.md                    Project overview
│   ├── getting-started.md          Installation, first run, experiment workflow
│   └── architecture.md             Pipeline design, module responsibilities, data flow
│
└── research/                 ← INTERNAL DOCS — for the team
    ├── experiments/                 Experiment reports (dated, reproducible)
    │   └── TEMPLATE.md              Standard template for new reports
    ├── plans/                       Research plans & proposals
    ├── meetings/                    Meeting notes
    └── references/                  Theory & literature
        ├── problem-setup.md         DRO circuit discovery problem formulation
        └── experiment-setup.md      ERM vs DRO experimental protocol
```

## Which section should I read?

| I want to ...                              | Go to            |
|--------------------------------------------|------------------|
| Install and run the project                | `user/getting-started.md` |
| Understand the pipeline and modules        | `user/architecture.md` |
| Read the formal problem setup              | `research/references/problem-setup.md` |
| Read the experiment design (ERM vs DRO)    | `research/references/experiment-setup.md` |
| Read an experiment report                  | `research/experiments/` |
| Check what was discussed in a meeting      | `research/meetings/` |

## Building the user docs locally

```bash
pip install mkdocs-material mkdocstrings[python]
cd docs/
mkdocs serve        # live preview at http://127.0.0.1:8000
mkdocs build        # static site output to site/
```

## Writing conventions

**User docs (`user/`)** are rendered by MkDocs and may be published. Write for an external reader who has ML background but no prior knowledge of this project.

**Research docs (`research/`)** are not rendered by MkDocs. They stay in the repo for team reference. Follow these conventions:

- **Experiments:** Name files as `YYYY-MM-DD_short-description.md`. Copy `TEMPLATE.md` to start. Every report must include: goal, environment, exact commands, results, and conclusion.
- **Meetings:** Name files as `YYYY-MM-DD.md`.
- **References:** Free-form, but include source links and a date header.
