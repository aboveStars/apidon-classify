import tensorflow as tf

def predict(news):
    # Load
    model = tf.keras.models.load_model('/Users/ali/Documents/Apidon/TextClassification/trained_model')

    # Class names
    class_names = ['World', 'Sports', 'Business', 'Sci/Tech']

    # Convert
    news = tf.convert_to_tensor(news)

    # Predict the news type
    preds = model.predict(news)
    pred_class = tf.argmax(preds[0]).numpy()

    print(f'Predicted class name: {class_names[pred_class]}')

if __name__ == "__main__":
    news = ['In the last weeks, there has been many transfer suprises in footbal. Ronaldo went back to Old Trafford,"while Messi went to Paris Saint Germain to join his former colleague Neymar.We cant wait to see these two clubs will perform in upcoming leagues']
    predict(news)
