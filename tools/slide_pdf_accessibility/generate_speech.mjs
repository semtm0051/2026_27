import fs from 'node:fs/promises';
import sre from 'speech-rule-engine';
const [input,output] = process.argv.slice(2);
if (!input || !output) throw new Error('Usage: node generate_speech.mjs input.json output.json');
await sre.setupEngine({locale:'en',domain:'mathspeak',style:'default',markup:'none'});
await sre.engineReady();
const data=JSON.parse(await fs.readFile(input,'utf8'));
const map={};
for (const equation of data.equations) {
  const speech=sre.toSpeech(equation.mathml);
  if (!speech.trim()) throw new Error('No English speech for '+equation.hash);
  map[equation.hash]=speech;
}
await fs.writeFile(output,JSON.stringify(map,null,2));
console.log('Generated English descriptions for '+Object.keys(map).length+' distinct equations.');
