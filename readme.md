## .NET 7 and ML.NET

This is a reinterpreted sample project based on the [Microsoft sentiment analysis tutorial][docs].

It used .NET 7 to load yelp reviews from a text file, train a model, test its accuracy, and then allow you to pass in sentiment to predict a result.

This project uses Spectre.Console to create a repeating input prompt and result set.

## Notes

- The dataset is not very large, about 1000 rows. That means the accuracy is going to be suspect.
- Training on 1000 records is fast, but you can save the model to disk to avoid retraining on each restart. In this case, you don't get a lot of performance benefits.

## Experience

- As a developer, and not as a ML scientist, the API for ML.NET can be confusing at first. The `MLContext` type is a work store, so you end up doing your work in that context.
- The context holds data, but it also has helpers for trainers, loading data, testing. This can feel strange to C# developers, but maybe not ML folks.
- Once you have a prediction engine, it all makes sense. Pass in a similar piece of data to your training data, get a prediction back.
- It's a powerful library and can do what you need to build, save, reload, and use models.


[docs]: https://learn.microsoft.com/en-us/dotnet/machine-learning/tutorials/sentiment-analysis