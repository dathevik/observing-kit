# Observing Kit

A collection of Python tools and scripts for astronomical observation follow-up work, including downloading image stamps, creating finding charts, and analyzing quasar spectra.

## Repository Structure

This repository is organized into modules, each containing specialized tools for different aspects of observational astronomy:

- **`QSO_followup/`** - Tools for QSO (Quasi-Stellar Object) follow-up observations
  - Download optical and radio image stamps from various surveys
  - Generate finding charts with accurate WCS coordinates
  - Create radio-optical overlay visualizations
  - Plot quasar spectra

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/dathevik/observing-kit.git
   cd observing-kit
   ```

2. **Install dependencies**:
   ```bash
   pip install astropy matplotlib numpy reproject
   ```

3. **Navigate to the module you need** and check its README for detailed usage instructions.

## Detailed Documentation

For detailed usage instructions, requirements, and examples, please refer to the README files in each directory:

- **[QSO_followup/README.md](QSO_followup/README.md)** - Complete documentation for QSO follow-up observation tools

## Requirements

- Python 3.6+
- `astropy` - For FITS file handling and WCS operations
- `matplotlib` - For plotting and visualization
- `numpy` - For numerical operations
- `reproject` - For image reprojection (required for radio-optical overlays)

## License

[Add your license information here]

## Contact

[Add contact information if desired]

