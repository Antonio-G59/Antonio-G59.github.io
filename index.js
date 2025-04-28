async function runExample() {
    let inputIds = [
        'console', 'alcohol_reference', 'animated_blood', 'blood', 'blood_and_gore',
        'cartoon_violence', 'crude_humor', 'drug_reference', 'fantasy_violence', 'intense_violence',
        'language', 'lyrics', 'mature_humor', 'mild_blood', 'mild_cartoon_violence',
        'mild_fantasy_violence', 'mild_language', 'mild_lyrics', 'mild_suggestive_themes',
        'mild_violence', 'no_descriptors', 'nudity', 'partial_nudity', 'sexual_content',
        'sexual_themes', 'simulated_gambling', 'strong_language', 'strong_sexual_content',
        'suggestive_themes', 'use_of_alcohol', 'use_of_drugs_and_alcohol', 'violence'
    ];

    let x = new Float32Array(32);

    for (let i = 0; i < inputIds.length; i++) {
        let element = document.getElementById(inputIds[i]);
        if (element) {
            x[i] = parseFloat(element.value) || 0;
        } else {
            console.error(`Missing element with id '${inputIds[i]}'`);
            x[i] = 0;
        }
    }

    let tensorX = new onnx.Tensor(x, 'float32', [1, 32]);

    let session = new onnx.InferenceSession();
    await session.loadModel("./DLnet_gamerating.onnx");

    let outputMap = await session.run({ input: tensorX });
    let outputData = outputMap.get('output1');

    let predictions = document.getElementById('predictions');

    let ratings = ["E", "ET", "T", "M"];
    let ratingIndex = Math.round(outputData.data[0]);
    if (ratingIndex < 0 || ratingIndex >= ratings.length) {
        ratingIndex = 0;
    }

    predictions.innerHTML = `
        <hr> Game Rating Prediction: <br/>
        <table>
            <tr>
                <td>Predicted Rating</td>
                <td id="td0">${ratings[ratingIndex]}</td>
            </tr>
        </table>`;
}
