#!/bin/bash
# Copy necessary files to the tests/data directory
# If files not present in tests/data/* then copy
if [ ! -f tests/data/mnist_cnn.pt ]; then
    echo "Copying mnist_cnn.pt to tests/data/"
    cp A4S_metric_tests/mnist_cnn.pt tests/data/mnist_cnn.pt 
fi

if [ ! -f tests/data/income_model.pt ]; then
    echo "Copying income_model.pt to tests/data/"
    cp A4S_metric_tests/income_model.pt tests/data/income_model.pt 
fi

if [ ! -f tests/data/metric_testing_dataset.pkl ]; then
    echo "Copying metric_testing_dataset.pkl to tests/data/"
    cp A4S_metric_tests/metric_testing_dataset.pkl tests/data/metric_testing_dataset.pkl 
fi

if [ ! -f tests/data/income_metric_testing_dataset.pkl ]; then
    echo "Copying income_metric_testing_dataset.pkl to tests/data/"
    cp A4S_metric_tests/income_metric_testing_dataset.pkl tests/data/income_metric_testing_dataset.pkl
fi

if [ ! -f tests/data/mnist_cnn.pt ]; then
    echo "Copying mnist_cnn.pt to tests/data/"
    cp A4S_metric_tests/mnist_cnn.pt tests/data/mnist_cnn.pt
fi

# Run the tests
uv run pytest tests/metrics/model_metrics -v