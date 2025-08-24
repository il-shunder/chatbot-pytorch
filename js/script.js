window.parent.document.displayMessages = function (message, answer) {
    const messagesBlock = window.parent.document.getElementById("messages");

    if (messagesBlock) {
        const messageElement = window.parent.document.createElement("div");
        messageElement.innerHTML = `
            <div class="user-message">${message}</div>
            <div class="chatbot-message">${answer}</div>
        `;

        messagesBlock.appendChild(messageElement);
    }
};
