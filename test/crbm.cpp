#include "catch.hpp"

#include "dll/conv_rbm.hpp"
#include "dll/conv_layer.hpp"
#include "dll/vector.hpp"
#include "dll/generic_trainer.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"


TEST_CASE( "crbm/mnist_1", "crbm::simple" ) {
    dll::conv_rbm<dll::conv_layer<
            28, 12, 40,
            dll::batch_size<25>
            >> rbm;

    auto dataset = mnist::read_dataset<std::vector, vector, double>();

    REQUIRE(!dataset.training_images.empty());
    dataset.training_images.resize(100);

    mnist::binarize_dataset(dataset);

    rbm.learning_rate = 0.001;
    auto error = rbm.train(dataset.training_images, 100);

    REQUIRE(error < 2e-3);
}