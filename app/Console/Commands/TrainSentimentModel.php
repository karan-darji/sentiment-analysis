<?php

namespace App\Console\Commands;

use Dom\Text;
use Illuminate\Console\Command;
use Rubix\ML\Classifiers\MultilayerPerceptron;
use Rubix\ML\Classifiers\NaiveBayes;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Layers\BatchNorm;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\PReLU;
use Rubix\ML\NeuralNet\Optimizers\AdaMax;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Pipeline;
use Rubix\ML\Transformers\TextNormalizer;
use Rubix\ML\Transformers\TfIdfTransformer;
use Rubix\ML\Transformers\WordCountVectorizer;
use Rubix\ML\Transformers\ZScaleStandardizer;

class TrainSentimentModel extends Command
{
    /**
     * The name and signature of the console command.
     *
     * @var string
     */
    protected $signature = 'train-sentiment';

    /**
     * The console command description.
     *
     * @var string
     */
    protected $description = 'This command is used to train a model for sentiment analysis using the provided dataset.';

    /**
     * Execute the console command.
     */
    public function handle()
    {
        $this->info('Starting sentiment model training...');
        $this->info('Loading dataset...');
        $dataset = Labeled::fromIterator(new CSV(storage_path('app/sentiment.csv')));

        $samples = ['Good'];
        $labels = ['Positive'];

        $newset = new Labeled($samples, $labels);
        // dd($newset);

        // dd($dataset);

        $this->info('Training model...');
        $estimator = new PersistentModel(
            new Pipeline(
                [
                    new TextNormalizer(),
                    new WordCountVectorizer(),
                    new TfIdfTransformer(),
                    new ZScaleStandardizer()
                ],
                new MultilayerPerceptron(
                    [
                        new Dense(100),
                        new Activation(new LeakyReLU()),
                        new Dense(100),
                        new Activation(new LeakyReLU()),
                        new Dense(100, 0.0, false),
                        new BatchNorm(),
                        new Activation(new LeakyReLU()),
                        new Dense(50),
                        new PReLU(),
                        new Dense(50),
                        new PReLU()
                    ],
                    256,
                    new AdaMax(0.001)
                )
            ),
            new Filesystem(storage_path('app/sentiment_model.rbx'))
        );
        $estimator->train($dataset);

        $this->info('saving model...');
        $estimator->save();
        $this->info('Model trained and saved successfully to ' . storage_path('app/sentiment_model.rbx'));
    }
}
