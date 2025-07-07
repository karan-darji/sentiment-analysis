<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

class SentimentValidation extends Command
{
    /**
     * The name and signature of the console command.
     *
     * @var string
     */
    protected $signature = 'sentiment-validation';

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
        $dataset = Labeled::fromIterator(new CSV(storage_path('app/sentiment.csv')))->randomize()->take(10);

        $estimator = PersistentModel::load(new Filesystem(storage_path('app/sentiment_model.rbx')));

        $this->info('Validating model with 10 random samples from the dataset...');

        $predictions = $estimator->predict($dataset);

        $report = new AggregateReport([
            new MulticlassBreakdown(),
            new ConfusionMatrix()
        ]);

        $result = $report->generate($predictions,$dataset->labels());
        $this->info('Validation Report:');
        $this->info($result->toJSON());

    }
}
