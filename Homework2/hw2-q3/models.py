import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


def reshape_state(state):
    h_state = state[0]
    c_state = state[1]
    new_h_state = torch.cat([h_state[:-1], h_state[1:]], dim=2)
    new_c_state = torch.cat([c_state[:-1], c_state[1:]], dim=2)
    return (new_h_state, new_c_state)


class BahdanauAttention(nn.Module):
    """
    Bahdanau attention mechanism:
    score(h_i, s_j) = v^T * tanh(W_h h_i + W_s s_j)
    """

    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()

        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)  # Transform encoder hidden states
        self.W_s = nn.Linear(hidden_size, hidden_size, bias=False)  # Transform decoder state
        self.v = nn.Linear(hidden_size, 1, bias=False)  # Scoring function

    def forward(self, query, encoder_outputs, src_lengths):
        """
        query: (batch_size, 1, hidden_size) - Decoder hidden state at current timestep
        encoder_outputs: (batch_size, max_src_len, hidden_size) - Encoder outputs
        src_lengths: (batch_size) - Lengths of input sequences

        Returns:
            attn_out: (batch_size, 1, hidden_size) - Context vector (weighted sum of encoder outputs)
        """

        # Compute attention scores
        scores = self.v(torch.tanh(self.W_h(encoder_outputs) + self.W_s(query)))  # (batch_size, max_src_len, 1)

        # Apply mask for padded sequences
        mask = self.sequence_mask(src_lengths).unsqueeze(-1)  # (batch_size, max_src_len, 1)
        scores = scores.masked_fill(~mask, -1e9)  # Mask out padded values

        # Convert scores to probabilities using softmax
        attn_weights = torch.softmax(scores, dim=1)  # (batch_size, max_src_len, 1)

        # Compute context vector as weighted sum of encoder outputs
        attn_out = torch.sum(attn_weights * encoder_outputs, dim=1, keepdim=True)  # (batch_size, 1, hidden_size)

        return attn_out

    def sequence_mask(self, lengths):
        """
        Creates a boolean mask from sequence lengths.
        True for valid positions, False for padding.
        """
        batch_size = lengths.numel()
        max_len = lengths.max()
        return (torch.arange(max_len, device=lengths.device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .lt(lengths.unsqueeze(1)))


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        hidden_size,
        padding_idx,
        dropout,
    ):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size // 2
        self.dropout = dropout

        self.embedding = nn.Embedding(
            src_vocab_size,
            hidden_size,
            padding_idx=padding_idx,
        )
        self.lstm = nn.LSTM(
            hidden_size,
            self.hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(self.dropout)

    def forward(
        self,
        src,
        lengths,
    ):
        # src: (batch_size, max_src_len)
        # lengths: (batch_size)
        #############################################
        # TODO: Implement the forward pass of the encoder

        # Hints:
        # - Use torch.nn.utils.rnn.pack_padded_sequence to pack the padded sequences
        #   (before passing them to the LSTM)
        # - Use torch.nn.utils.rnn.pad_packed_sequence to unpack the packed sequences
        #   (after passing them to the LSTM)
        #############################################
        embedded = self.dropout(self.embedding(src))
        packed_embedded = pack(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        encoder_outputs, _ = unpack(packed_output, batch_first=True)
        return encoder_outputs, (hidden, cell)
        #############################################
        # END OF YOUR CODE
        #############################################
        # enc_output: (batch_size, max_src_len, hidden_size)
        # final_hidden: tuple with 2 tensors
        # each tensor is (num_layers * num_directions, batch_size, hidden_size)


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        tgt_vocab_size,
        attn,
        padding_idx,
        dropout,
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.tgt_vocab_size = tgt_vocab_size
        self.attn = attn  # Bahdanau Attention Layer

        self.embedding = nn.Embedding(
            tgt_vocab_size, hidden_size, padding_idx=padding_idx
        )

        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_size * 2, hidden_size)  # Combines LSTM output with context vector

    def forward(self, tgt, dec_state, encoder_outputs, src_lengths):
        """
        tgt: (batch_size, max_tgt_len) - Target sequences
        dec_state: Tuple (h, c) - Decoder state
        encoder_outputs: (batch_size, max_src_len, hidden_size) - Encoder outputs
        src_lengths: (batch_size) - Lengths of source sequences

        Returns:
            - outputs: (batch_size, max_tgt_len, hidden_size) - Decoder outputs
            - dec_state: Updated decoder state
        """

        if dec_state[0].shape[0] == 2:  # If bidirectional encoder, reshape state
            dec_state = reshape_state(dec_state)

        embedded = self.dropout(self.embedding(tgt))  # (batch_size, max_tgt_len, hidden_size)

        outputs = []
        for t in range(tgt.size(1)):  # Loop over each timestep
            lstm_input = embedded[:, t].unsqueeze(1)  # (batch_size, 1, hidden_size)
            lstm_out, dec_state = self.lstm(lstm_input, dec_state)  # (batch_size, 1, hidden_size)

            if self.attn is not None:  # Apply attention if enabled
                context = self.attn(lstm_out, encoder_outputs, src_lengths)  # (batch_size, 1, hidden_size)
                lstm_out = torch.cat((lstm_out, context), dim=2)  # Concatenate LSTM output with context
                lstm_out = torch.tanh(self.fc_out(lstm_out))  # (batch_size, 1, hidden_size)

            outputs.append(lstm_out)

        outputs = torch.cat(outputs, dim=1)  # (batch_size, max_tgt_len, hidden_size)
        return outputs, dec_state


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = nn.Linear(decoder.hidden_size, decoder.tgt_vocab_size)
        self.generator.weight = self.decoder.embedding.weight

    def forward(self, src, src_lengths, tgt, dec_hidden=None):
        encoder_outputs, final_enc_state = self.encoder(src, src_lengths)

        if dec_hidden is None:
            dec_hidden = final_enc_state

        # **Fix: Only pass tgt[:, :-1] to decoder to avoid EOS mismatch**
        output, dec_hidden = self.decoder(
            tgt[:, :-1], dec_hidden, encoder_outputs, src_lengths
        )

        # **Fix: Ensure output shape matches target for loss computation**
        output = self.generator(output)

        return output, dec_hidden
