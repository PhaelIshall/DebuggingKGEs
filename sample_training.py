import pandas as pd
from pykeen.pipeline import pipeline
from pykeen.datasets.freebase import FB15k237

def main():
    dataset = FB15k237()
    result = pipeline(
        training=dataset.training,
        validation=dataset.validation,
        testing=dataset.testing,
        model='RotatE',
        model_kwargs = dict(
            embedding_dim= 1000,
            entity_initializer= "uniform",
            relation_initializer= "init_phases",
            relation_constrainer="complex_normalize"
        ),
        optimizer="Adam",
        optimizer_kwargs=dict(
            lr= 0.00005
        ),
        loss= "nssa",
        loss_kwargs= dict(
            reduction= "mean",
            adversarial_temperature= 1.0,
            margin=9
        ),
        training_loop = "SLCWA",
        negative_sampler= "basic",
        negative_sampler_kwargs = dict(
            num_negs_per_pos= 256
        ),
        training_kwargs = dict(
            num_epochs= 1000,
            batch_size= 1024
        ),
        evaluator_kwargs = dict(
        filtered= True
        )
    )
    metrics = result.metric_results.to_df()
    training = result.training
    # MRR: The inverse of the harmonic mean over all ranks.
    # MR: The arithmetic mean over all ranks.
    
    tail_results= metrics[metrics['Side']=='tail']
    tail_results_realistic = tail_results[tail_results['Type'] == 'realistic']
    print(tail_results_realistic)
    mr = tail_results_realistic[tail_results_realistic['Metric']=='arithmetic_mean_rank']['Value'].values[0]
    
    mrr = tail_results_realistic[tail_results_realistic['Metric']=='inverse_harmonic_mean_rank']['Value'].values[0]
    hits_at_1 = tail_results_realistic[tail_results_realistic['Metric']=='hits_at_1']['Value'].values[0]
    hits_at_5 = tail_results_realistic[tail_results_realistic['Metric']=='hits_at_5']['Value'].values[0]
    hits_at_10 = tail_results_realistic[tail_results_realistic['Metric']=='hits_at_10']['Value'].values[0]
    
    print("----------------RESULTS----------------")
    print(f'MR: {mr}, MRR: {mrr}, Hits@1: {hits_at_1}, Hits@5: {hits_at_5}, Hits@10: {hits_at_10}')


    # Get predictions for each testing triple:  
    # predictions_df = model.get_all_prediction_df(triples_factory=dataset.training, testing=dataset.testing)

main()