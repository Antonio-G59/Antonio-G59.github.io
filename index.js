
async function runExample() {
  let x = [];

  const fieldIds = [
      'console', 'alcohol_reference', 'animated_blood', 'blood', 'blood_and_gore',
      'cartoon_violence', 'crude_humor', 'drug_reference', 'fantasy_violence', 'intense_violence',
      'language', 'lyrics', 'mature_humor', 'mild_blood', 'mild_cartoon_violence',
      'mild_fantasy_violence', 'mild_language', 'mild_lyrics', 'mild_suggestive_themes',
      'mild_violence', 'no_descriptors', 'nudity', 'partial_nudity', 'sexual_content',
      'sexual_themes', 'simulated_gambling', 'strong_janguage', 'strong_sexual_content',
      'suggestive_themes', 'use_of_alcohol', 'use_of_drugs_and_alcohol', 'violence'
  ];

  for (let i = 0; i < fieldIds.length; i++) {
    const element = document.getElementById(fieldIds[i]);
    if (!element) {
      console.error(`Missing input with id: ${fieldIds[i]}`);
      return;
    }
    const val = parseFloat(element.value);
    x.push(val);
  }

  let floatArray = new Float32Array(x.map(v => parseFloat(v)));
  let tensorX = new ort.Tensor('float32', floatArray, [1, 33]);
  let session = await ort.InferenceSession.create('./DLnet_video_game.onnx');


  let outputMap = await session.run({ input: tensorX });
  let outputData = outputMap.values().next().value;

  let predictions = document.getElementById('predictions');
  predictions.innerHTML = `
  <hr>Got an output tensor:<br/>
  <pre>${JSON.stringify(outputData.data, null, 2)}</pre>
  <table>
      <tr>
          <td>Game rating</td>
          <td>${outputData.data[0].toFixed(2)}</td>
      </tr>
  </table>
`;

}
