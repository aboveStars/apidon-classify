import tensorflow as tf

def predict(news):
    # Load
    model = tf.keras.models.load_model('/content/text_model22.keras')

    # Class names
    class_names = ['World', 'Sports', 'Business', 'Sci/Tech']

    # Convert
    news = tf.convert_to_tensor(news)

    # Predict the news type
    preds = model.predict(news)
    pred_class = tf.argmax(preds[0]).numpy()

    print(f'Predicted class name: {class_names[pred_class]}')

if __name__ == "__main__":
    news = ['Tech Giants Report Record Profits as Global Markets Rebound; Investors Optimistic About Economic Recovery Despite Supply Chain Challenges and Inflation Concerns Looming on the Horizon']
    predict(news)
