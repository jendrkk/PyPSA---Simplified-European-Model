---
description: 'Project specific instructions for Python code in the PyPSA Simplified European Model'
applyTo: '**/*.py'
---

# Project Specific Instructions

## General Idea

- This project aims to be a proof of concept of the model pypsa-eur.
- This project is a simplified version of the PyPSA-Eur model, focusing on key aspects while reducing complexity.
- The backbone of the model is PyPSA (Python for Power System Analysis), an open-source tool for simulating and optimizing power systems.
- Modules and functions should be methodologically equal or similar to those in pypsa-eur where possible.

## Structure

- src/pypsa_simplified: Contains the main source code for the simplified PyPSA model analogous to pypsa-eur.
- scripts: Contains scripts for running simulations, data processing, and other tasks.
- notebooks: Contains Jupyter notebooks for data exploration, analysis, and visualization and use the methods from src/pypsa_simplified.
- data: Contains datasets used in the model, including raw, processed and geometry data.

## Approach

- Implement methods and functions that mirror those in pypsa-eur, adapting them to the simplified context.
- Always ensure that you use pypsa as the core library for power system analysis.
- Always evaluate using existing implementations in pypsa-eur before creating new ones.
- Consult the sources listed below for reference and guidance.
- Propose improvements or simplifications that maintain the integrity of the original model.
- Propose next steps or additional features that could enhance the simplified model and follow from pypsa-eur.

## Useful Sources

- [PyPSA GitHub Repository](https://github.com/PyPSA/PyPSA.git)
- [PyPSA Documentation](https://pypsa.org/doc/)
- [PyPSA-Eur GitHub Repository](https://github.com/PyPSA/pypsa-eur.git)
- [PyPSA-Eur Documentation](https://pypsa-eur.readthedocs.io/en/latest/)