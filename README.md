[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Build Status](https://img.shields.io/badge/PINN-yes-green)
![Build Status](https://img.shields.io/badge/FDM-yes-blue)
![Contributions](https://img.shields.io/badge/contributions-welcome-gold)
![GitHub issues](https://img.shields.io/github/issues/DennisWayo/3D-PINN-FDM-FracEnergyBalance)
![GitHub forks](https://img.shields.io/github/forks/DennisWayo/3D-PINN-FDM-FracEnergyBalance)
![GitHub stars](https://img.shields.io/github/stars/DennisWayo/3D-PINN-FDM-FracEnergyBalance)


## Extending Physics-Informed Neural Networks for Error-Corrected Finite Difference Energy Balance Modeling in 3D Hydraulic Fracture Propagation

This repository contains the **Physics-Informed Neural Network (PINN)-enhanced Finite Difference Method (FDM)** implementation for modeling **energy balance in hydraulic fracture propagation**. The study explores **1D, 2D, and 3D formulations** to improve numerical accuracy and computational efficiency in fracture simulations.


### Methodology
The PINN-enhanced FDM framework integrates physics constraints into deep learning models to correct numerical errors in finite difference approximations. The key steps include:

1. **Synthetic Data Generation**: Simulating fracture scenarios under varying pressure gradients, viscosity, and dimensions.
2. **Finite Difference Approximation**: Discretizing the energy balance equation for numerical solutions.
3. **PINN Training**: Learning correction terms using physics-informed loss functions.
4. **Performance Evaluation**: Assessing model accuracy via **Mean Squared Error (MSE)** and **error reduction metrics**.

### Results
- **MSE Reduction**: 
  - **1D PINN**: 1.0179  
  - **2D PINN**: 0.3653 (Best performance)  
  - **3D PINN**: 0.7593  
- **Optimal Training Setup**: Batch size **128**, 3-layer network.
- **Computational Efficiency**: PINN reduces oscillatory behavior near fracture tips while maintaining stability.

![pinn_comparison_animation (2)](https://github.com/user-attachments/assets/d4f4b678-c846-4713-b4fa-22d7a7d6fe36)

### Running the Code
To train the **1D, 2D, and 3D PINN models**, execute:

```bash
python code/2&3D_FracEnergy.ipynb
```
For visualization and evaluation, use the **Google Colab platform**.

### Citation
If you use this work, please cite:

```
@article{your_paper_reference,
  title={Extending Physics-Informed Neural Networks for Error-Corrected Finite Difference Energy Balance Modeling in 3D Hydraulic Fracture Propagation},
  author={xx},
  journal={xx},
  year={2025},
  doi={xx}
}
```

### Funders
- The National Key R&D Program of China (2023YFE0110900)
- National Natural Science Foundation of China (52074040)
- Nazarbayev University’s Collaborative Research Project (111024CRP2014)

### Future Work
- Extension to **fully coupled fluid-solid interaction models**.
- Adaptive PINN training for complex **geomechanical applications**.
- Integration with high-performance computing (HPC) frameworks.

### Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

### Contact
For questions, reach out via email (iwayoden@gmail.com) or GitHub discussions.
