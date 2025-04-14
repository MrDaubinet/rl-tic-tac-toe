<script>
  // Data Props
  export let score = [0, 0, 0]
  export let agent_play = false
</script>

<div class="min-w-md">
  <div class="flex justify-between text-2xl">
    <div>
      {score[0]} 
      <span class="ml-2">
        {#if agent_play}
          Player
        {:else}
          Player 1
        {/if}
      </span>
    </div>
    <div>
      {score[2]}  <span class="ml-2">Ties</span>
    </div>
    <div>
      {score[1]} <span class="ml-2 ">
        {#if agent_play}
          Agent
        {:else}
          Player 2
        {/if}
      </span>
    </div>
  </div>
</div>