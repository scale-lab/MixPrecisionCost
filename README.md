# MixPC: Mixed Precision Cost Estimator

The **Mixed Precision Cost Estimator** is a powerful utility package designed for machine learning practitioners and researchers to accurately assess the computational cost of deep learning models based on mixed precision. This package reads a model’s structure and calculates the FLOPs (Floating Point Operations) and MACs (Multiply-Accumulate Operations) for each layer and module, while also extracting the weight and activation precisions. By providing a custom cost function (such as MACs * weight_precision² * activation_precision), the package estimates the cost for every layer and module, summing them up to deliver a total cost for the entire model. This innovative tool enables precise cost estimation for models using mixed precision, offering a new metric to benchmark and optimize neural networks for efficiency. The package is compatible with all PyTorch models and is currently applied in multi-task learning contexts. It is particularly valuable for automating the search for efficient mixed precision configurations, driving both performance and resource optimization in model design.

## Installation

```bash
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps mixpc_mboulila
```

## Usage

```python
from mixpc import estimate_cost

model = ... # your model
input_shape = ... # your input shape
f = ... # your cost function

cost, model_info = estimate_cost(model, input_shape, f)
```

where `model_info` is a dictionary containing the model's structure and the cost of each layer and module, and `cost` is the total cost of the model.

to know more about how to use the package, you can check the [example](https://github.com/mboulila/MixPrecisionCost/blob/main/examples/example.ipynb)

to know what to put in the `input_shape` argument, you can check the [ptflops](https://github.com/sovrasov/flops-counter.pytorch) documentation.

to get an idea of what to put in the `f` argument, you can check the [cost function](#cost-function) section.

## Cost function

The `f` argument is a function that takes a dictionary of the model's structure and returns the cost of the model.

The dictionary has the following structure:

```python
{
    "name": [
    "module": "module of the layer",
    "macs": "number of macs of the layer",
    "params": "number of parameters of the layer",
    "cost": "cost of the layer"
    ]
}
```

for example, if you want to use the MACs * weight_precision² * activation_precision cost function, you can do the following:

```python
f = lambda x,y,z: x*y**2*z
```
A good default cost function is the MACs * weight_precision * activation_precision, which is the [ACE: Arithmetic Computation Effort](https://openaccess.thecvf.com/content/CVPR2024/papers/Neseem_PikeLPN_Mitigating_Overlooked_Inefficiencies_of_Low-Precision_Neural_Networks_CVPR_2024_paper.pdf) Metric.


## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

For any questions or suggestions, please contact the author at [mahdi_boulila@brown.edu](mailto:mahdi_boulila@brown.edu).

## Acknowledgments

This package uses the `ptflops` library to calculate the model's complexity.
This package was inspired by the need to accurately estimate the computational cost of deep learning models using mixed precision.
This package is developed under the supervision of Professor [Sherief Reda](https://faculty.brown.edu/cogsci/people/david-cox/) within the [ScaLe Lab](https://scale-lab.github.io/) at Brown University.