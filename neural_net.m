function err = neural_net(hidden_layers)

% Training hyperparameters
NUM_EPOCHS = 5;
TRAINING_SET_SIZE = 60000;
TEST_SET_SIZE = 10000;
MINI_BATCH_SIZE = 100;
MINI_BATCHES_IN_TRAINING_SET = TRAINING_SET_SIZE / MINI_BATCH_SIZE;

WEIGHT_INIT_MEAN = 0;
WEIGHT_INIT_STDDEV = 0.01;

LEARNING_RATE = 0.01;
WEIGHT_DECAY = 0;

NUM_INPUTS = 784;
NUM_OUTPUTS = 10;

% Neural net layer sizes
LAYERS = [NUM_INPUTS hidden_layers NUM_OUTPUTS];
NUM_LAYERS = columns(LAYERS)-1;

% Weight and bias cells
weights = cell(NUM_LAYERS);
weight_grads = cell(NUM_LAYERS);
biases = cell(NUM_LAYERS);
bias_grads = cell(NUM_LAYERS);
relu_bits = cell(NUM_LAYERS);
layer_input = cell(NUM_LAYERS);

% Weight and bias initialization
for l = 1:NUM_LAYERS
	weights{l} = normrnd(WEIGHT_INIT_MEAN, WEIGHT_INIT_STDDEV, LAYERS(l+1), LAYERS(l));
	biases{l} = zeros(LAYERS(l+1), 1);
end

% Load training data
train_imgs_file = fopen("train_images.gz", "rzb");
train_imgs = fread(train_imgs_file);
train_imgs = train_imgs(17:end);	% Discard file header

train_lbls_file = fopen("train_labels.gz", "rzb");
train_lbls = fread(train_lbls_file);
train_lbls = train_lbls(9:end);		% Discard file header

% Training
for epoch = 1:NUM_EPOCHS
	train_imgs_mb_start = 1;
	train_imgs_mb_end = NUM_INPUTS * MINI_BATCH_SIZE;
	train_lbls_mb_start = 1;
	train_lbls_mb_end = MINI_BATCH_SIZE;

	for mb = 1:MINI_BATCHES_IN_TRAINING_SET

		% Fetch images in mini-batch
		imgs = train_imgs(train_imgs_mb_start:train_imgs_mb_end);
		imgs = reshape(imgs, NUM_INPUTS, MINI_BATCH_SIZE);
		train_imgs_mb_start = train_imgs_mb_end + 1;
		train_imgs_mb_end = train_imgs_mb_start + NUM_INPUTS * MINI_BATCH_SIZE - 1;

		% Fetch labels in mini-batch
		lbls = train_lbls(train_lbls_mb_start:train_lbls_mb_end);
		lbls = reshape(lbls, 1, MINI_BATCH_SIZE);
		train_lbls_mb_start = train_lbls_mb_end + 1;
		train_lbls_mb_end = train_lbls_mb_start + MINI_BATCH_SIZE - 1;

		% Forward pass
		for l = 1:NUM_LAYERS
			layer_input{l} = imgs;
			imgs = weights{l} * imgs + (biases{l} * ones(1, MINI_BATCH_SIZE));

			% ReLU
			relu_bits{l} = (imgs > 0);
			imgs = max(imgs, 0);
		end

		% Softmax
		expimgs = exp(imgs);
		imgs = expimgs ./ (ones(NUM_OUTPUTS, 1) * sum(expimgs));

		% Cross-entropy loss
		identity = eye(NUM_OUTPUTS);
		onehotlbls = identity(1:end, lbls+1);
		loss = sum(-log(sum(imgs .* onehotlbls))) / MINI_BATCH_SIZE;

		% printf ("Epoch %d, mini-batch %d - loss = %f\n", epoch, mb, loss);

		% Initial gradient
		imgs = imgs - onehotlbls;

		% Backward pass
		for l = 1:NUM_LAYERS
			m = NUM_LAYERS+1 - l;

			% ReLU backpropagation
			imgs = imgs .* relu_bits{m};

			% Gradient compute
			weight_grads{m} = imgs * layer_input{m}';
			bias_grads{m} = sum(imgs')';

			% Layer backpropagation
			imgs = weights{m}' * imgs;
		end

		% Gradient descent
		for l = 1:NUM_LAYERS
			weights{l} = weights{l} - ((LEARNING_RATE / MINI_BATCH_SIZE) * weight_grads{l} + WEIGHT_DECAY * weights{l}); 
			biases{l} = biases{l} - ((LEARNING_RATE / MINI_BATCH_SIZE) * bias_grads{l} + WEIGHT_DECAY * biases{l});
		end		
	end
end

% Load test data
test_imgs_file = fopen("test_images.gz", "rzb");
test_imgs = fread(test_imgs_file);
test_imgs = test_imgs(17:end);	% Discard file header
test_imgs = reshape(test_imgs, NUM_INPUTS, TEST_SET_SIZE);

test_lbls_file = fopen("test_labels.gz", "rzb");
test_lbls = fread(test_lbls_file);
test_lbls = test_lbls(9:end);	% Discard file header
test_lbls = reshape(test_lbls, 1, TEST_SET_SIZE);

% Testing
for l = 1:NUM_LAYERS
	test_imgs = weights{l} * test_imgs + (biases{l} * ones(1, TEST_SET_SIZE));

	% ReLU
	test_imgs = max(test_imgs, 0);
end

% Softmax
expimgs = exp(test_imgs);
test_imgs = expimgs ./ (ones(NUM_OUTPUTS, 1) * sum(expimgs));

% Accuracy
[mxs, inds] = max(test_imgs);
err = 1 - (sum((test_lbls+1) == inds) / TEST_SET_SIZE);

% printf("Testing done - final error: %f\n", err);
endfunction