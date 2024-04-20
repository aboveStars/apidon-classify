import tensorflow as tf

def predict(sample_news):
    # Load the trained model
    model = tf.keras.models.load_model('/Users/ali/Documents/Apidon/TextClassification/trained_model')

    # Define the class names
    class_names = ['World', 'Sports', 'Business', 'Sci/Tech']

    # Convert sample news into array
    sample_news = tf.convert_to_tensor(sample_news)

    # Predict the news type
    preds = model.predict(sample_news)

    pred_class = tf.argmax(preds[0]).numpy()

    print(f'Predicted class: {pred_class} \nPredicted class name: {class_names[pred_class]}')

if __name__ == "__main__":
    sample_news = ['In the last weeks, there has been many transfer suprises in footbal. Ronaldo went back to Old Trafford,"while Messi went to Paris Saint Germain to join his former colleague Neymar.We cant wait to see these two clubs will perform in upcoming leagues']
    predict(sample_news)

