
// Randomly selects one of my favorite quotes for display
function chooseQuote() {

    // Define set of quotes
    const quotes = ["\"Science is not only a disciple of reason, but also one of romance and passion.\" &emsp;- Stephen Hawking",
                    "\"Science can never solve one problem without creating ten more.\" &emsp;- George Bernard Shaw",
                    "\"Now is the time to understand more, so that we may fear less.\" &emsp;- Marie Curie",
                    "\"Nothing is static, nothing is final, everything is held provisionally.\" &emsp;- Jocelyn Bell Burnell",
                    "\"What is intelligible is also beautiful.\" &emsp;- Subrahmanyan Chandrasekhar",
                    "\"I used to think that science would save us, and science certainly tried.\" &emsp;- Kurt Vonnegut",
                    "\"Science progresses best when observations force us to alter our preconceptions.\" &emsp;- Vera Rubin",
                    "\"What\'s the use of doing all this work if we don't get some fun out of this?\" &emsp;- Rosalind Franklin",
                    "\"Only mathematics and mathematical logic can say as little as the physicist means to say.\" &emsp;- Bertrand Russell",
                    "\"We are progeny of the cosmos and our ability to understand it is an inheritance.\" &emsp;- Janna Levin"];

    // Select a random quote to display
    const nQuotes = quotes.length;
    const rQuoteID = Math.floor(Math.random()*nQuotes);
    var qem = document.getElementById('quote');
    qem.innerHTML += quotes[rQuoteID];
}