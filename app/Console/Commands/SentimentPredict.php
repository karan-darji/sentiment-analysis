<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use League\CommonMark\Normalizer\TextNormalizer;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

class SentimentPredict extends Command
{
    /**
     * The name and signature of the console command.
     *
     * @var string
     */
    protected $signature = 'sentiment-predict';

    /**
     * The console command description.
     *
     * @var string
     */
    protected $description = 'Command description';

    /**
     * Execute the console command.
     */
    public function handle()
    {
        $estimator = PersistentModel::load(new Filesystem(storage_path('app/sentiment_model.rbx')));
        $comment = $this->ask('Enter a comment to predict sentiment:');

        $dataset = new Unlabeled([[$comment]]);

        $prediction = $estimator->predict($dataset);
        $this->info('Predicted sentiment: ' . $prediction[0]);
    }
}
