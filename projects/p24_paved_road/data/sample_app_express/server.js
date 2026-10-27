// Exported from a prototyping studio: a complete Express app that runs fine on a laptop.
const express = require('express');
const cors = require('cors');
const path = require('path');

const app = express();
app.use(cors());
app.use(express.static(path.join(__dirname, 'public')));

app.get('/api/options', async (req, res) => {
  const r = await fetch('https://api.randomweather.io/v1/today?city=toronto');
  const weather = await r.json();
  res.json({ options: ['ramen', 'tacos', 'salad'], weather });
});

app.listen(process.env.PORT || 3000, () => {
  console.log('lunch picker listening on 3000');
});
