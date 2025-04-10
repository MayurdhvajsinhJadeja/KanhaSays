# KanhaSays

KanhaSays is an AI-powered system that delivers verses from the Bhagavad Gita, aiding motivation. Built with Python and libraries like NLTK, TensorFlow, and Flask, it employs NLP and neural networks to interpret user queries, providing relevant verses. Users can translate or listen to verses and switch between light and dark themes for customization. This is helpful to the people who are suffering from depression or are frustrated with something. It Provides inspiration and guidance to individuals facing various challenges, including depression, frustration, and anxiety. Its ability to deliver personalized verses from the Bhagavad Gita, along with translations and audio options, empowers users to find solace and motivation in the timeless wisdom of this sacred text. Whether it's students grappling with exam stress or individuals navigating life's complexities, it offers a beacon of hope and encouragement, helping them find inner strength and resilience amidst adversity. They need to input their personal thoughts and they will get motivated from the quotes of Bhagvat Gita with their meanings.

## Requirements

- Python 3.x
- Natural Language Toolkit (NLTK)
- Tensorflow
- Pandas
- NumPy
- keras
- flask

## Usage

The website will ask you to enter your query, and then provide you with a relevant verse from the Bhagavad Gita. You can also choose to translate the verse into your preferred language or listen to it. Additionally, you can switch between dark and light themes based on your preference.

To run the app, go to the deployment folder and then run the app.py file using command `python app.py` 

The app will be started at `https://127.0.0.1:5000`

## Detailed Explanation

KanhaSays works by making use of a JSON file that contains all of the verses from the Bhagavad Gita. We use natural language processing (NLP) and neural network techniques to train the model on the JSON file. When a user enters a query, the model converts the query into tokens and intents, and then searches the JSON file for the relevant verse. The system provides the user with the verse and also allows them to translate it and listen to it.

## Screenshots

- Light Mode
- ![LightMode](https://github.com/MayurdhvajsinhJadeja/KanhaSays/blob/main/lightmode.jpg)

- Translation
- ![Translation](https://github.com/MayurdhvajsinhJadeja/KanhaSays/blob/main/translated.jpg)

- Dark Mode
- ![DarkMode](https://github.com/MayurdhvajsinhJadeja/KanhaSays/blob/main/darkmode.jpg)

## Future Work

In the future, we plan to improve the system by using a semantic approach to better understand the user's intent and provide more accurate verses. We also plan to add a chatbot feature that will remember the user's previous inputs and provide more personalized responses. Additionally, we plan to allow users to create their own accounts and save their favorite verses for easy access.


## Contributors

- Mayurdhvajsinh Jadeja (https://github.com/MayurdhvajsinhJadeja)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
