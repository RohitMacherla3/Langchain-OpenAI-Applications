css = '''
<style>
.chat-message {
    padding: 1.5rem; 
    border-radius: 0.5rem; 
    margin-bottom: 1rem; 
    display: flex;
    align-items: flex-start;
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
    width: 20%;
    min-width: 78px;
    max-width: 78px;
    margin-right: 1rem;
}

.chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    width: 80%;
    padding: 0 1rem;
    color: #fff;
    word-wrap: break-word;
    overflow-wrap: break-word;
}

/* Fix for mobile responsiveness */
@media (max-width: 768px) {
    .chat-message .avatar {
        width: 60px;
        min-width: 60px;
        max-width: 60px;
    }
    .chat-message .avatar img {
        max-width: 60px;
        max-height: 60px;
    }
    .chat-message .message {
        width: calc(100% - 80px);
        padding: 0 0.5rem;
    }
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/zNYyrQz/MACHERLA-R-1615-5.jpg" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''