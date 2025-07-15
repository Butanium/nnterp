# NNsight Glossary

- **Dispatch**: By default, NNsight do not load the weights of LanguageModel and initialize it with FakeTensors, which allow you to use `.scan()` where you can check the dimensions of the tensors returned by different modules. When you do trace however, the model's weights needs to be loaded. This is the "dispatch".