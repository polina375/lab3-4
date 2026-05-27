#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "neuralnet.h"
#include "trainer.h"

namespace py = pybind11;

PYBIND11_MODULE(nn_core, m)
{
    m.doc() = "Neural Network for Laboratory works 3-4";

    py::class_<Neural::NeuralNetwork<float>>(m, "NeuralNetwork")
        .def(py::init<int, int>(),
            py::arg("input_size") = 2,
            py::arg("hidden_size") = 16)

        .def("forward", &Neural::NeuralNetwork<float>::forward)
        .def("predict_class", &Neural::NeuralNetwork<float>::predictClass,
            py::arg("features"),
            py::arg("threshold") = 0.5f)

        .def("save", &Neural::NeuralNetwork<float>::save)
        .def("load", &Neural::NeuralNetwork<float>::load)

        .def("get_input_size", &Neural::NeuralNetwork<float>::getInputSize)
        .def("get_hidden_size", &Neural::NeuralNetwork<float>::getHiddenSize);

    m.def("train_model",
        &Neural::Trainer::train<float>,
        py::arg("net"),
        py::arg("features"),
        py::arg("labels"),
        py::arg("epochs") = 1000,
        py::arg("learning_rate") = 0.1f);
}